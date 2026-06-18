"""AutoAnalyst — demo server.

A FastAPI app that serves the analyst UI and runs the agent. Pick a bundled
dataset (or, when running locally, upload your own CSV/Excel), ask a question,
and get back the full step-by-step trace the agent produced plus its answer.

Run locally:
    uvicorn app.server:app --reload --port 8000
Then open http://localhost:8000

Each request gets its own fresh sandbox, so unlike a single-GPU model server
there is no global lock — analyses are independent.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env (GROQ_API_KEY) if present — harmless in production where the env is
# already populated (e.g. an HF Space secret).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from autoanalyst.agent import DEFAULT_MAX_STEPS, DEFAULT_MODEL, Analyst
from autoanalyst.dataio import SAMPLES, load_table, sample_registry

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

# --- config (env-overridable so HF Spaces / CI can swap behaviour) --------------
SAMPLES_DIR = Path(os.environ.get("AUTOANALYST_SAMPLES_DIR", str(ROOT / "samples")))
UPLOAD_DIR = Path(os.environ.get("AUTOANALYST_UPLOAD_DIR", str(ROOT / "uploads")))
ALLOW_UPLOAD = os.environ.get("AUTOANALYST_ALLOW_UPLOAD", "1") == "1"
MODEL = os.environ.get("AUTOANALYST_MODEL", DEFAULT_MODEL)
MAX_STEPS = int(os.environ.get("AUTOANALYST_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
MAX_UPLOAD_MB = 10

UPLOADS: dict[str, Path] = {}  # session_id -> uploaded file path (in-process)

app = FastAPI(title="AutoAnalyst", description="Agentic data-analysis demo")


# --- helpers -------------------------------------------------------------------
def _sample_path(dataset_id: str) -> Path | None:
    meta = SAMPLES.get(dataset_id)
    if not meta:
        return None
    p = SAMPLES_DIR / meta["file"]
    return p if p.exists() else None


def _resolve_data(dataset_id: str | None, session_id: str | None):
    """Return (df, schema_summary, label) for a sample id or an upload session."""
    if dataset_id:
        path = _sample_path(dataset_id)
        if not path:
            raise HTTPException(404, f"unknown dataset '{dataset_id}'")
        df, schema = load_table(path)
        return df, schema, SAMPLES[dataset_id]["label"]
    if session_id:
        if not ALLOW_UPLOAD:
            raise HTTPException(403, "uploads are disabled on this deployment")
        path = UPLOADS.get(session_id)
        if not path or not path.exists():
            raise HTTPException(404, "upload session not found — please re-upload")
        df, schema = load_table(path)
        return df, schema, path.stem
    raise HTTPException(400, "provide a dataset_id or an upload session_id")


# --- API -----------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    question: str
    dataset_id: str | None = None
    session_id: str | None = None


@app.get("/api/health")
def health():
    return {
        "status": "online" if os.environ.get("GROQ_API_KEY") else "offline",
        "model": MODEL,
        "allow_upload": ALLOW_UPLOAD,
        "reason": None if os.environ.get("GROQ_API_KEY") else "GROQ_API_KEY not set",
    }


@app.get("/api/datasets")
def datasets():
    return [d for d in sample_registry() if _sample_path(d["id"])]


@app.get("/api/schema")
def schema(dataset_id: str):
    df, schema_text, label = _resolve_data(dataset_id, None)
    return {
        "dataset_id": dataset_id,
        "label": label,
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        "row_count": int(df.shape[0]),
        "preview": df.head(8).astype(object).where(df.head(8).notna(), None)
        .values.tolist(),
        "preview_columns": list(df.columns),
        "schema_text": schema_text,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not ALLOW_UPLOAD:
        raise HTTPException(403, "uploads are disabled on this deployment")
    suffix = Path(file.filename or "data.csv").suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(400, "please upload a .csv or .xlsx file")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (limit {MAX_UPLOAD_MB} MB)")

    UPLOAD_DIR.mkdir(exist_ok=True)
    session_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{session_id}{suffix}"
    path.write_bytes(data)
    UPLOADS[session_id] = path

    try:
        df, schema_text = load_table(path)
    except Exception as exc:  # noqa: BLE001 — bad file -> clean 400
        path.unlink(missing_ok=True)
        UPLOADS.pop(session_id, None)
        raise HTTPException(400, f"could not read that file: {exc}")

    return {
        "session_id": session_id,
        "label": Path(file.filename or "your data").stem,
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        "row_count": int(df.shape[0]),
        "preview": df.head(8).astype(object).where(df.head(8).notna(), None)
        .values.tolist(),
        "preview_columns": list(df.columns),
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.question.strip():
        raise HTTPException(400, "a question is required")
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(503, "model offline — GROQ_API_KEY is not set")

    df, schema_text, label = _resolve_data(req.dataset_id, req.session_id)
    analyst = Analyst(df, schema_text, model=MODEL, max_steps=MAX_STEPS)
    try:
        result = analyst.run(req.question)
    except Exception as exc:  # noqa: BLE001 — surface upstream/model errors cleanly
        raise HTTPException(502, f"analysis failed: {exc}")
    return {"label": label, **result.to_dict()}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/api/analyze/stream")
def analyze_stream(question: str, dataset_id: str | None = None,
                   session_id: str | None = None):
    """Server-Sent Events: emits each step as the agent produces it, then the
    final answer — so the UI can render the analysis live."""
    if not question.strip():
        raise HTTPException(400, "a question is required")
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(503, "model offline — GROQ_API_KEY is not set")

    df, schema_text, label = _resolve_data(dataset_id, session_id)
    analyst = Analyst(df, schema_text, model=MODEL, max_steps=MAX_STEPS)

    def gen():
        yield _sse({"type": "start", "label": label})
        try:
            for kind, payload in analyst.run_iter(question):
                if kind == "step":
                    yield _sse({"type": "step", **payload.to_dict()})
                else:
                    yield _sse({"type": "final", "label": label, **payload.to_dict()})
        except Exception as exc:  # noqa: BLE001 — stream a clean error event
            yield _sse({"type": "error", "detail": f"analysis failed: {exc}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --- static frontend (mounted last so /api/* wins) ------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC)), name="static")
