// AutoAnalyst frontend — vanilla JS, no framework.
const $ = (sel) => document.querySelector(sel);
const state = { datasetId: null, sessionId: null, datasets: [], busy: false };

// ---- status pill (polls /api/health) ----------------------------------------
async function pollHealth() {
  const el = $("#status");
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    const online = h.status === "online";
    el.className = "status " + (online ? "online" : "offline");
    el.querySelector(".label").textContent = online ? "Online" : "Offline";
    el.title = online ? `model: ${h.model}` : (h.reason || "offline");
    if (h.allow_upload) $("#uploadZone").classList.remove("hidden");
  } catch {
    el.className = "status offline";
    el.querySelector(".label").textContent = "Offline";
  }
}

// ---- datasets ----------------------------------------------------------------
async function loadDatasets() {
  const grid = $("#datasetGrid");
  try {
    state.datasets = await fetch("/api/datasets").then((r) => r.json());
  } catch {
    grid.innerHTML = `<p class="muted">Could not load datasets.</p>`;
    return;
  }
  grid.innerHTML = "";
  state.datasets.forEach((d) => {
    const card = document.createElement("button");
    card.className = "ds-card";
    card.innerHTML = `<div class="ds-name">${esc(d.label)}</div>
                      <div class="ds-desc">${esc(d.description)}</div>`;
    card.onclick = () => selectDataset(d, card);
    grid.appendChild(card);
  });
}

async function selectDataset(d, card) {
  document.querySelectorAll(".ds-card").forEach((c) => c.classList.remove("active"));
  card.classList.add("active");
  state.datasetId = d.id;
  state.sessionId = null;
  renderChips(d.questions || []);
  $("#analyzeBtn").disabled = false;
  await loadSchema({ dataset_id: d.id });
}

async function loadSchema(params) {
  const panel = $("#schemaPanel");
  const url = params.dataset_id
    ? `/api/schema?dataset_id=${encodeURIComponent(params.dataset_id)}`
    : null;
  let s;
  try {
    s = url ? await fetch(url).then((r) => r.json()) : params.uploaded;
  } catch {
    return;
  }
  $("#schemaLabel").textContent = s.label || "";
  $("#schemaMeta").textContent =
    `${s.row_count.toLocaleString()} rows · ${s.columns.length} columns`;
  const head = s.preview_columns || s.columns.map((c) => c.name);
  const rows = (s.preview || []).slice(0, 6);
  const dtypes = Object.fromEntries(s.columns.map((c) => [c.name, c.dtype]));
  let html = "<thead><tr>" +
    head.map((c) => `<th>${esc(c)}<br><span class="dtype">${esc(dtypes[c] || "")}</span></th>`).join("") +
    "</tr></thead><tbody>";
  rows.forEach((r) => {
    html += "<tr>" + r.map((v, i) =>
      `<td class="${i === 0 ? "col-name" : ""}">${esc(fmt(v))}</td>`).join("") + "</tr>";
  });
  html += "</tbody>";
  $("#schemaTable").innerHTML = html;
  panel.classList.remove("hidden");
}

// ---- upload (local only) -----------------------------------------------------
$("#uploadBtn") && ($("#uploadBtn").onclick = () => $("#fileInput").click());
$("#fileInput") && ($("#fileInput").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append("file", file);
  try {
    const r = await fetch("/api/upload", { method: "POST", body });
    if (!r.ok) throw new Error((await r.json()).detail || "upload failed");
    const s = await r.json();
    state.sessionId = s.session_id;
    state.datasetId = null;
    document.querySelectorAll(".ds-card").forEach((c) => c.classList.remove("active"));
    renderChips([]);
    $("#analyzeBtn").disabled = false;
    await loadSchema({ uploaded: s });
  } catch (err) {
    showBanner(`Upload failed: ${err.message}`);
  }
});

// ---- example chips -----------------------------------------------------------
function renderChips(questions) {
  const wrap = $("#chips");
  wrap.innerHTML = "";
  questions.forEach((q) => {
    const c = document.createElement("button");
    c.className = "chip";
    c.textContent = q;
    c.onclick = () => { $("#question").value = q; analyze(); };
    wrap.appendChild(c);
  });
}

// ---- analyze -----------------------------------------------------------------
$("#analyzeBtn").onclick = analyze;
$("#question").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
});

let stream = null;
function analyze() {
  if (state.busy) return;
  const question = $("#question").value.trim();
  if (!question) return;
  if (!state.datasetId && !state.sessionId) {
    showBanner("Pick a dataset (or upload one) first.");
    return;
  }
  setBusy(true);
  const results = $("#results");
  results.innerHTML = "";
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.innerHTML = `<span class="orb"></span><span class="t-label">AutoAnalyst is reading the data…</span>`;
  const setThinking = (t) => thinking.querySelector(".t-label").textContent = t;
  results.appendChild(thinking);

  const params = new URLSearchParams({ question });
  if (state.datasetId) params.set("dataset_id", state.datasetId);
  else params.set("session_id", state.sessionId);

  let finished = false;
  const done = () => { finished = true; thinking.remove(); stream.close(); setBusy(false); };
  stream = new EventSource(`/api/analyze/stream?${params}`);

  stream.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "step") {
      results.insertBefore(renderStep(msg), thinking);
      setThinking(`Ran step ${msg.n} — deciding the next move…`);
    } else if (msg.type === "final") {
      done();
      results.appendChild(renderAnswer(msg));
    } else if (msg.type === "error") {
      done();
      showBanner(msg.detail || "analysis failed");
    }
  };
  stream.onerror = () => {
    if (finished) return;
    done();
    showBanner("Connection lost while analyzing. Is the model online?");
  };
}

function renderResult(data) {
  const results = $("#results");
  results.innerHTML = "";
  (data.trace || []).forEach((step) => results.appendChild(renderStep(step)));
  results.appendChild(renderAnswer(data));
}

function renderStep(step) {
  const el = document.createElement("div");
  el.className = "step";
  let inner = `<div class="step-head">
      <span class="step-badge">${step.n}</span>
      <span class="step-tool"><b>run_python</b> · step ${step.n}</span>
    </div>
    <pre class="code">${highlight(step.code || "")}</pre>`;
  const io = [];
  if (step.stdout && step.stdout.trim())
    io.push(`<div class="io-block"><div class="io-label">Output</div>
      <pre class="io-out">${esc(step.stdout.trim())}</pre></div>`);
  if (step.error)
    io.push(`<div class="io-block"><div class="io-label">Error</div>
      <pre class="io-out err">${esc(step.error.trim())}</pre></div>`);
  if (io.length) inner += `<div class="io">${io.join("")}</div>`;
  (step.charts || []).forEach((c) => {
    inner += `<div class="chart-wrap"><img alt="chart from step ${step.n}"
      src="data:image/png;base64,${c}"></div>`;
  });
  el.innerHTML = inner;
  return el;
}

function renderAnswer(data) {
  const el = document.createElement("div");
  el.className = "answer";
  const meta = data.stopped === "step_limit"
    ? `reached the ${data.steps_used}-step limit`
    : `${data.steps_used} step${data.steps_used === 1 ? "" : "s"}`;
  el.innerHTML = `<div class="a-head"><span class="a-mark">▶</span>
      <span class="a-title">Answer</span></div>
    <div class="a-body">${esc(data.answer || "(no answer)")}</div>
    <div class="a-meta">${esc(data.label || "")} · ${meta}</div>`;
  return el;
}

// ---- helpers -----------------------------------------------------------------
function setBusy(b) {
  state.busy = b;
  const btn = $("#analyzeBtn");
  btn.classList.toggle("busy", b);
  btn.disabled = b;
}
function showBanner(msg) {
  const results = $("#results");
  const b = document.createElement("div");
  b.className = "banner-err";
  b.innerHTML = `<b>⚠</b> ${esc(msg)}`;
  results.prepend(b);
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(2);
  return v;
}

// minimal Python syntax highlighter — single-pass tokenizer so HTML entities
// are never re-processed (each token is escaped exactly once before wrapping).
const PY_KW = new Set(["import", "from", "as", "def", "return", "if", "elif",
  "else", "for", "while", "in", "not", "and", "or", "is", "None", "True",
  "False", "lambda", "with", "print"]);
const PY_BI = new Set(["df", "pd", "np", "plt", "sum", "mean", "count", "groupby",
  "sort_values", "head", "describe", "value_counts", "plot", "title", "xlabel",
  "ylabel", "len", "round", "sorted", "agg", "merge", "pivot_table"]);
function highlight(code) {
  const re = /(#[^\n]*)|('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const [full, com, str, num, ident] = m;
    if (com) out += `<span class="com">${esc(com)}</span>`;
    else if (str) out += `<span class="str">${esc(str)}</span>`;
    else if (num) out += `<span class="num">${num}</span>`;
    else if (PY_KW.has(ident)) out += `<span class="kw">${ident}</span>`;
    else if (PY_BI.has(ident)) out += `<span class="bi">${ident}</span>`;
    else out += esc(ident);
    last = re.lastIndex;
  }
  out += esc(code.slice(last));
  return out;
}

// ---- boot --------------------------------------------------------------------
pollHealth();
setInterval(pollHealth, 15000);
loadDatasets();
