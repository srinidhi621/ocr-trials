#!/usr/bin/env python3
"""
Lightweight local UI for running the OCR pipeline.
"""

import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_file, abort, render_template_string

from main import run_pipeline, generate_run_id

BASE_OUTPUT_DIR = Path("./output")
RUNS_DIR = BASE_OUTPUT_DIR

app = Flask(__name__)

_runs_lock = threading.Lock()
_runs: Dict[str, Dict[str, Any]] = {}


HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Document OCR Pipeline</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; max-width: 900px; }
      .row { margin-bottom: 12px; }
      label { display: inline-block; width: 160px; }
      input[type="text"], input[type="number"], select { width: 200px; }
      #log { white-space: pre-wrap; background: #f6f6f6; padding: 12px; border: 1px solid #ddd; height: 240px; overflow: auto; }
      .muted { color: #666; }
      .links a { display: inline-block; margin-right: 12px; }
    </style>
  </head>
  <body>
    <h2>Document OCR Pipeline</h2>
    <div class="row muted">Local-only UI. Outputs stay under ./output.</div>
    <form id="run-form">
      <div class="row">
        <label>PDF File</label>
        <input type="file" name="pdf" accept="application/pdf" required />
      </div>
      <div class="row">
        <label>Provider</label>
        <select name="provider">
          <option value="azure" selected>azure</option>
          <option value="vertex">vertex</option>
        </select>
      </div>
      <div class="row">
        <label>DPI</label>
        <input type="number" name="dpi" value="300" min="150" max="600" />
      </div>
      <div class="row">
        <label>Enhance</label>
        <input type="checkbox" name="enhance" checked />
      </div>
      <div class="row">
        <label>Save Artifacts</label>
        <input type="checkbox" name="save_artifacts" />
      </div>
      <div class="row">
        <label>Signature Report</label>
        <input type="checkbox" name="signature_report" checked />
      </div>
      <div class="row">
        <button type="submit">Run Pipeline</button>
      </div>
    </form>

    <hr/>
    <div class="row"><strong>Run ID:</strong> <span id="run-id">-</span></div>
    <div class="row"><strong>Status:</strong> <span id="status">-</span></div>
    <div class="row"><strong>Output Directory:</strong> <span id="output-dir">-</span></div>
    <div class="row links" id="output-links"></div>

    <h3>Logs</h3>
    <div id="log">Waiting for run...</div>

    <script>
      const form = document.getElementById('run-form');
      const runIdEl = document.getElementById('run-id');
      const statusEl = document.getElementById('status');
      const outputDirEl = document.getElementById('output-dir');
      const outputLinksEl = document.getElementById('output-links');
      const logEl = document.getElementById('log');
      let pollHandle = null;
      let currentRunId = null;

      function setOutputLinks(outputs) {
        outputLinksEl.innerHTML = '';
        if (!outputs || outputs.length === 0) return;
        outputs.forEach(o => {
          const a = document.createElement('a');
          a.href = o.url;
          a.textContent = o.label;
          a.target = '_blank';
          outputLinksEl.appendChild(a);
        });
      }

      function setLogText(text) {
        const isAtBottom = Math.abs(logEl.scrollHeight - logEl.clientHeight - logEl.scrollTop) < 8;
        logEl.textContent = text;
        if (isAtBottom) {
          logEl.scrollTop = logEl.scrollHeight;
        }
      }

      async function pollStatus() {
        if (!currentRunId) return;
        const res = await fetch(`/status/${currentRunId}`);
        const data = await res.json();
        statusEl.textContent = data.status;
        outputDirEl.textContent = data.output_dir || '-';
        setLogText(data.log || 'No logs yet.');
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollHandle);
          pollHandle = null;
          if (data.status === 'completed') {
            const outputsRes = await fetch(`/outputs/${currentRunId}`);
            const outputs = await outputsRes.json();
            setOutputLinks(outputs.files || []);
          }
        }
      }

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        statusEl.textContent = 'starting';
        setLogText('Starting run...');
        outputLinksEl.innerHTML = '';

        const res = await fetch('/run', { method: 'POST', body: formData });
        const data = await res.json();
        currentRunId = data.run_id;
        runIdEl.textContent = data.run_id;
        statusEl.textContent = data.status;
        outputDirEl.textContent = data.output_dir || '-';

        if (pollHandle) clearInterval(pollHandle);
        pollHandle = setInterval(pollStatus, 2000);
      });
    </script>
  </body>
</html>
"""


def _read_log_tail(log_path: Path, max_lines: int = 200) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _safe_resolve(base_dir: Path, relative_path: str) -> Optional[Path]:
    try:
        target = (base_dir / relative_path).resolve()
    except Exception:
        return None
    base = base_dir.resolve()
    if not str(target).startswith(str(base)):
        return None
    return target


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template_string(HTML_PAGE)


@app.route("/run", methods=["POST"])
def run_pipeline_route():
    if "pdf" not in request.files:
        return jsonify({"error": "Missing PDF file"}), 400
    pdf_file = request.files["pdf"]
    if not pdf_file.filename:
        return jsonify({"error": "Invalid PDF filename"}), 400

    provider = request.form.get("provider", "azure").lower()
    dpi = int(request.form.get("dpi", 300))
    enhance = "enhance" in request.form
    save_artifacts = "save_artifacts" in request.form
    signature_report = "signature_report" in request.form

    run_id = generate_run_id(pdf_file.filename, provider)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / pdf_file.filename
    pdf_file.save(str(pdf_path))

    log_path = run_dir / f"{run_id}.log"

    with _runs_lock:
        _runs[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "pdf_path": str(pdf_path),
            "output_dir": str(run_dir),
            "log_path": str(log_path),
            "output_paths": {},
            "error": None,
        }

    def _worker():
        with _runs_lock:
            _runs[run_id]["status"] = "running"
        try:
            result = run_pipeline(
                pdf_path=str(pdf_path),
                provider=provider,
                dpi=dpi,
                enhance=enhance,
                output_dir=str(RUNS_DIR),
                save_artifacts=save_artifacts,
                signature_report=signature_report,
                verbose=False,
                run_id=run_id,
                show_console=False,
            )
            with _runs_lock:
                _runs[run_id]["status"] = "completed"
                _runs[run_id]["output_paths"] = result.get("output_paths", {})
        except Exception as exc:
            try:
                log_path.write_text(
                    log_path.read_text(encoding="utf-8", errors="ignore")
                    + f"\nERROR: {exc}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            with _runs_lock:
                _runs[run_id]["status"] = "failed"
                _runs[run_id]["error"] = str(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    return jsonify({
        "run_id": run_id,
        "status": "queued",
        "output_dir": str(run_dir),
    })


@app.route("/status/<run_id>", methods=["GET"])
def status(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    log_path = Path(run["log_path"])
    return jsonify({
        "run_id": run_id,
        "status": run["status"],
        "output_dir": run["output_dir"],
        "log": _read_log_tail(log_path),
        "error": run.get("error"),
    })


@app.route("/outputs/<run_id>", methods=["GET"])
def outputs(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    run_dir = Path(run["output_dir"])
    output_paths = run.get("output_paths", {})
    files = []
    for label, path in output_paths.items():
        if not path:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        rel_path = path_obj.relative_to(run_dir)
        files.append({
            "label": label,
            "url": f"/files/{run_id}/{rel_path.as_posix()}",
            "path": str(path_obj),
        })

    # Include signatures directory link if it exists
    signatures_dir = run_dir / "signatures"
    if signatures_dir.exists():
        files.append({
            "label": "signatures_dir",
            "url": f"/files/{run_id}/signatures/",
            "path": str(signatures_dir),
        })

    return jsonify({
        "run_id": run_id,
        "files": files,
    })


@app.route("/files/<run_id>/<path:requested_path>", methods=["GET"])
def files(run_id: str, requested_path: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if not run:
        abort(404)
    run_dir = Path(run["output_dir"])
    target = _safe_resolve(run_dir, requested_path)
    if not target:
        abort(400)
    if target.is_dir():
        # Basic directory listing for signatures folder
        entries = [p.name for p in target.iterdir() if p.is_file()]
        return jsonify({
            "directory": str(target),
            "files": entries,
        })
    if not target.exists():
        abort(404)
    return send_file(target)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
