"""A small, persistent Python execution sandbox for agent-written code.

The agent's `run_python` tool calls land here. The sandbox holds one namespace
(so `df` and intermediate variables survive across steps), captures stdout and
any matplotlib figure the code draws, and applies a pragmatic safety check.

The safety check (AST-based denylist + a soft timeout) is deliberately modest,
not bulletproof: the public demo only ever runs against curated, bundled
datasets inside a non-root container, which is what actually contains the risk.
"""
from __future__ import annotations

import ast
import base64
import io
import threading
import traceback
from contextlib import redirect_stdout

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAX_OUTPUT = 6000  # chars of stdout fed back to the model (keeps context sane)
DEFAULT_TIMEOUT = 20.0  # seconds per snippet

# Imports we never allow agent code to make.
_BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "urllib2", "urllib3", "httpx", "http", "importlib", "ctypes",
    "multiprocessing", "pickle", "marshal", "glob", "pathlib", "builtins",
    "webbrowser", "ftplib", "smtplib", "telnetlib", "pty",
}
# Builtins we never allow agent code to call by name.
_BLOCKED_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "input", "exit", "quit",
    "globals", "locals", "memoryview",
}
# Dangerous attribute calls (e.g. os.system) even if the module slipped through.
_BLOCKED_ATTRS = {"system", "popen", "remove", "unlink", "rmtree", "spawn",
                  "fork", "kill", "chmod", "chown"}


def safety_check(code: str) -> str | None:
    """Return a reason string if the code is disallowed, else None.

    A SyntaxError is *not* blocked here — we let it run so the model sees the
    real error and corrects itself.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BLOCKED_MODULES:
                    return f"import of '{alias.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _BLOCKED_MODULES:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
                return f"call to '{func.id}()' is not allowed"
            if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_ATTRS:
                return f"call to '.{func.attr}()' is not allowed"
    return None


class SandboxResult:
    """Outcome of one snippet: text output, an error string, and base64 PNGs."""

    __slots__ = ("stdout", "error", "charts")

    def __init__(self, stdout: str, error: str | None, charts: list[str]):
        self.stdout = stdout
        self.error = error
        self.charts = charts


class PythonSandbox:
    """A persistent exec namespace seeded with the DataFrame and the usual tools."""

    def __init__(self, df: "pd.DataFrame"):
        self.ns: dict = {"pd": pd, "np": np, "plt": plt, "df": df}

    def run(self, code: str, timeout: float = DEFAULT_TIMEOUT) -> SandboxResult:
        reason = safety_check(code)
        if reason:
            return SandboxResult("", f"BlockedError: {reason}", [])

        buf = io.StringIO()
        box: dict = {"error": None}

        def _work() -> None:
            try:
                with redirect_stdout(buf):
                    exec(compile(code, "<analysis>", "exec"), self.ns)
            except Exception:  # noqa: BLE001 — surface any error back to the model
                box["error"] = traceback.format_exc(limit=3).strip()

        t = threading.Thread(target=_work, daemon=True)
        t.start()
        t.join(timeout)

        charts = self._collect_charts()
        out = buf.getvalue()
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n... [output truncated to {MAX_OUTPUT} chars]"
        if t.is_alive():
            return SandboxResult(
                out, f"TimeoutError: code ran longer than {timeout:.0f}s", charts
            )
        return SandboxResult(out, box["error"], charts)

    @staticmethod
    def _collect_charts() -> list[str]:
        """Grab any open matplotlib figures as base64 PNGs, then clear them."""
        charts: list[str] = []
        for num in plt.get_fignums():
            fig = plt.figure(num)
            try:
                b = io.BytesIO()
                fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
                charts.append(base64.b64encode(b.getvalue()).decode("ascii"))
            except Exception:  # noqa: BLE001 — a bad figure shouldn't kill the step
                pass
        plt.close("all")
        return charts
