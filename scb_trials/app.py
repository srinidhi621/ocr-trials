#!/usr/bin/env python3
"""
OCR Pipeline Local UI

Single-page Flask application for running the OCR pipeline locally.
Provides PDF upload, configuration, progress tracking, and output viewing.
"""

import os
import re
import sys
import json
import time
import base64
import shutil
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from flask import Flask, request, jsonify, send_file, abort, Response, send_from_directory
import markdown

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import run_pipeline, generate_run_id

# Configuration
PORT = int(os.environ.get("PORT", 5001))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./output"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
STATE_FILE = Path(os.environ.get("STATE_FILE", "./.ui_state.json"))
STATIC_DIR = Path(__file__).parent / "static"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload


# ============================================================================
# Run State Management
# ============================================================================

@dataclass
class RunState:
    """State for a single pipeline run."""
    run_id: str
    status: str  # queued, running, completed, failed
    pdf_path: str
    output_dir: str
    provider: str
    dpi: int
    enhance: bool
    save_artifacts: bool
    signature_report: bool
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    output_paths: Optional[Dict[str, str]] = None
    elapsed_time: Optional[float] = None


class StateManager:
    """Manages run states with disk persistence."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.lock = threading.Lock()
        self._runs: Dict[str, RunState] = {}
        self._load()
    
    def _load(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                for run_id, run_data in data.items():
                    self._runs[run_id] = RunState(**run_data)
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
    
    def _save(self):
        """Save state to disk."""
        try:
            data = {run_id: asdict(run) for run_id, run in self._runs.items()}
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state file: {e}")
    
    def create_run(self, run_id: str, pdf_path: str, output_dir: str,
                   provider: str, dpi: int, enhance: bool,
                   save_artifacts: bool, signature_report: bool) -> RunState:
        """Create a new run in queued state."""
        with self.lock:
            run = RunState(
                run_id=run_id,
                status="queued",
                pdf_path=pdf_path,
                output_dir=output_dir,
                provider=provider,
                dpi=dpi,
                enhance=enhance,
                save_artifacts=save_artifacts,
                signature_report=signature_report,
                created_at=datetime.now().isoformat(),
            )
            self._runs[run_id] = run
            self._save()
            return run
    
    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get run state by ID."""
        with self.lock:
            return self._runs.get(run_id)
    
    def update_run(self, run_id: str, **kwargs):
        """Update run state fields."""
        with self.lock:
            if run_id in self._runs:
                run = self._runs[run_id]
                for key, value in kwargs.items():
                    if hasattr(run, key):
                        setattr(run, key, value)
                self._save()
    
    def get_all_runs(self) -> List[RunState]:
        """Get all runs, sorted by creation time (newest first)."""
        with self.lock:
            return sorted(
                self._runs.values(),
                key=lambda r: r.created_at,
                reverse=True
            )


# Global state manager
state_manager = StateManager(STATE_FILE)


# ============================================================================
# Background Pipeline Runner
# ============================================================================

def run_pipeline_background(run_id: str):
    """Run the pipeline in a background thread."""
    run = state_manager.get_run(run_id)
    if not run:
        return
    
    # Update status to running
    state_manager.update_run(run_id, status="running", started_at=datetime.now().isoformat())
    
    try:
        result = run_pipeline(
            pdf_path=run.pdf_path,
            provider=run.provider,
            dpi=run.dpi,
            enhance=run.enhance,
            output_dir=str(OUTPUT_DIR),
            save_artifacts=run.save_artifacts,
            signature_report=run.signature_report,
            verbose=True,
            run_id=run_id,
            show_console=False,
        )
        
        # Update with success
        state_manager.update_run(
            run_id,
            status="completed",
            completed_at=datetime.now().isoformat(),
            output_paths=result.get("output_paths"),
            elapsed_time=result.get("elapsed_time"),
        )
        
    except Exception as e:
        # Update with failure
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        state_manager.update_run(
            run_id,
            status="failed",
            completed_at=datetime.now().isoformat(),
            error=error_msg,
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.route("/")
def index():
    """Serve the main HTML page."""
    return HTML_TEMPLATE


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files."""
    return send_from_directory(STATIC_DIR, filename)


@app.route("/run", methods=["POST"])
def start_run():
    """Upload PDF and start pipeline run."""
    # Check for PDF file
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    pdf_file = request.files["pdf"]
    if pdf_file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF"}), 400
    
    # Get configuration options
    provider = request.form.get("provider", "azure")
    dpi = int(request.form.get("dpi", 300))
    enhance = request.form.get("enhance", "true").lower() == "true"
    save_artifacts = request.form.get("save_artifacts", "false").lower() == "true"
    signature_report = request.form.get("signature_report", "true").lower() == "true"
    
    # Validate DPI
    if not 150 <= dpi <= 600:
        return jsonify({"error": "DPI must be between 150 and 600"}), 400
    
    # Generate run ID
    run_id = generate_run_id(pdf_file.filename, provider)
    
    # Save uploaded PDF
    pdf_path = UPLOAD_DIR / f"{run_id}.pdf"
    pdf_file.save(str(pdf_path))
    
    # Create run state
    output_dir = str(OUTPUT_DIR / run_id)
    run = state_manager.create_run(
        run_id=run_id,
        pdf_path=str(pdf_path),
        output_dir=output_dir,
        provider=provider,
        dpi=dpi,
        enhance=enhance,
        save_artifacts=save_artifacts,
        signature_report=signature_report,
    )
    
    # Start pipeline in background thread
    thread = threading.Thread(target=run_pipeline_background, args=(run_id,), daemon=True)
    thread.start()
    
    return jsonify({
        "run_id": run_id,
        "status": run.status,
        "output_dir": output_dir,
    })


@app.route("/status/<run_id>")
def get_status(run_id: str):
    """Get status of a pipeline run."""
    run = state_manager.get_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    # Read log file if it exists
    log_content = ""
    log_path = OUTPUT_DIR / run_id / f"{run_id}.log"
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
        except Exception:
            log_content = "(Could not read log file)"
    
    # Calculate elapsed time if running
    elapsed = None
    if run.status == "running" and run.started_at:
        started = datetime.fromisoformat(run.started_at)
        elapsed = (datetime.now() - started).total_seconds()
    elif run.elapsed_time:
        elapsed = run.elapsed_time
    
    return jsonify({
        "run_id": run.run_id,
        "status": run.status,
        "log": log_content,
        "output_dir": run.output_dir,
        "error": run.error,
        "elapsed_time": elapsed,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    })


@app.route("/outputs/<run_id>")
def get_outputs(run_id: str):
    """Get list of output files for a run."""
    run = state_manager.get_run(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    output_path = OUTPUT_DIR / run_id
    if not output_path.exists():
        return jsonify({"files": []})
    
    files = []
    for path in output_path.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(output_path)
            file_info = {
                "name": path.name,
                "path": str(rel_path),
                "size": path.stat().st_size,
                "view_url": f"/view/{run_id}/{rel_path}",
                "download_url": f"/files/{run_id}/{rel_path}",
            }
            
            # Categorize file type and priority
            if "_extracted.md" in path.name:
                file_info["type"] = "markdown"
                file_info["category"] = "document"
                file_info["priority"] = 0
            elif "_signature_report.md" in path.name:
                file_info["type"] = "markdown"
                file_info["category"] = "signature"
                file_info["priority"] = 1
            elif path.suffix == ".md":
                file_info["type"] = "markdown"
                file_info["category"] = "other"
                file_info["priority"] = 2
            elif path.suffix == ".json":
                file_info["type"] = "json"
                file_info["category"] = "data"
                file_info["priority"] = 3
            elif path.suffix == ".log":
                file_info["type"] = "log"
                file_info["category"] = "log"
                file_info["priority"] = 4
            elif path.suffix in [".png", ".jpg", ".jpeg"]:
                file_info["type"] = "image"
                file_info["category"] = "image"
                file_info["priority"] = 5
            else:
                file_info["type"] = "file"
                file_info["category"] = "other"
                file_info["priority"] = 6
            
            files.append(file_info)
    
    # Sort by priority, then name
    files.sort(key=lambda f: (f.get("priority", 99), f["name"]))
    
    return jsonify({"files": files})


@app.route("/view/<run_id>/<path:file_path>")
def view_file(run_id: str, file_path: str):
    """Render markdown/JSON or serve file with appropriate display."""
    run = state_manager.get_run(run_id)
    if not run:
        abort(404)
    
    full_path = OUTPUT_DIR / run_id / file_path
    if not full_path.exists() or not full_path.is_file():
        abort(404)
    
    # Security: ensure path is within output directory
    try:
        full_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        abort(403)
    
    suffix = full_path.suffix.lower()
    
    if suffix == ".md":
        # Render markdown as HTML
        with open(full_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Determine if this is a signature report (to embed images)
        is_signature_report = "_signature_report" in full_path.name
        is_extracted_doc = "_extracted.md" in full_path.name
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "codehilite", "toc", "md_in_html"]
        )
        
        # For signature reports, embed signature images as base64
        if is_signature_report:
            html_content = embed_signature_images(html_content, OUTPUT_DIR / run_id)
        
        # Choose template based on content type
        if is_extracted_doc:
            template = DOCUMENT_TEMPLATE
        elif is_signature_report:
            template = SIGNATURE_TEMPLATE
        else:
            template = MARKDOWN_TEMPLATE
        
        download_link = f'<a href="/files/{run_id}/{file_path}" class="download-btn">Download Markdown</a>'
        
        return (template
                .replace("{{content}}", html_content)
                .replace("{{title}}", full_path.name)
                .replace("{{download_link}}", download_link))
    
    elif suffix == ".json":
        # Display JSON with syntax highlighting
        with open(full_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        try:
            # Pretty print JSON
            parsed = json.loads(json_content)
            json_content = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass
        return JSON_TEMPLATE.replace("{{content}}", json_content).replace("{{title}}", full_path.name)
    
    elif suffix == ".log":
        # Display log as preformatted text
        with open(full_path, "r", encoding="utf-8") as f:
            log_content = f.read()
        return LOG_TEMPLATE.replace("{{content}}", log_content).replace("{{title}}", full_path.name)
    
    elif suffix in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
        # Serve images directly
        return send_file(full_path, mimetype=f"image/{suffix[1:]}")
    
    else:
        # Serve other files for download
        return send_file(full_path, as_attachment=True)


def embed_signature_images(html_content: str, output_dir: Path) -> str:
    """Embed signature images as base64 in the HTML content."""
    signatures_dir = output_dir / "signatures"
    if not signatures_dir.exists():
        return html_content
    
    # Find all image references and embed them
    def replace_image(match):
        img_path = match.group(1)
        # Handle relative paths
        if img_path.startswith("signatures/"):
            full_img_path = output_dir / img_path
        else:
            full_img_path = signatures_dir / img_path
        
        if full_img_path.exists():
            try:
                with open(full_img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                suffix = full_img_path.suffix.lower()[1:]
                mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
                return f'<img src="data:{mime};base64,{img_data}" alt="Signature"'
            except Exception:
                pass
        return match.group(0)
    
    # Replace image src attributes
    html_content = re.sub(r'<img src="([^"]+)"', replace_image, html_content)
    
    return html_content


@app.route("/files/<run_id>/<path:file_path>")
def serve_file(run_id: str, file_path: str):
    """Serve raw file or directory listing."""
    run = state_manager.get_run(run_id)
    if not run:
        abort(404)
    
    full_path = OUTPUT_DIR / run_id / file_path
    
    # Security: ensure path is within output directory
    try:
        full_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        abort(403)
    
    if full_path.is_file():
        return send_file(full_path)
    
    elif full_path.is_dir():
        # Return directory listing as JSON
        items = []
        for item in sorted(full_path.iterdir()):
            rel_path = item.relative_to(OUTPUT_DIR / run_id)
            items.append({
                "name": item.name,
                "path": str(rel_path),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
                "url": f"/files/{run_id}/{rel_path}" if item.is_dir() else f"/view/{run_id}/{rel_path}",
            })
        return jsonify({"directory": str(file_path), "items": items})
    
    abort(404)


@app.route("/runs")
def list_runs():
    """List all pipeline runs."""
    runs = state_manager.get_all_runs()
    return jsonify({
        "runs": [asdict(r) for r in runs[:50]]  # Limit to 50 most recent
    })


@app.route("/how-it-works")
def how_it_works():
    """Serve the HOW_IT_WORKS.md content as HTML."""
    how_it_works_path = Path(__file__).parent / "HOW_IT_WORKS.md"
    
    if not how_it_works_path.exists():
        return jsonify({"error": "Documentation not found"}), 404
    
    try:
        with open(how_it_works_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code", "codehilite", "toc"]
        )
        
        return jsonify({"html": html_content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# HTML Templates
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Pipeline</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #475569;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Header with logo */
        .top-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border);
            background: var(--bg-secondary);
        }
        
        .logo-section {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        
        .logo-section img {
            height: 36px;
            width: auto;
        }
        
        .app-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .app-subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        /* Layout */
        .main-layout {
            display: flex;
            min-height: calc(100vh - 70px);
        }
        
        /* Left sidebar - pinned */
        .sidebar {
            width: 320px;
            flex-shrink: 0;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 1.25rem;
            overflow-y: auto;
            position: sticky;
            top: 0;
            height: calc(100vh - 70px);
        }
        
        .sidebar h2 {
            font-size: 0.95rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Compact upload area */
        .upload-area {
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 1.25rem;
        }
        
        .upload-area:hover, .upload-area.dragover {
            border-color: var(--accent);
            background: rgba(59, 130, 246, 0.1);
        }
        
        .upload-area.has-file {
            border-color: var(--success);
            background: rgba(34, 197, 94, 0.1);
        }
        
        .upload-area input[type="file"] {
            display: none;
        }
        
        .upload-icon {
            font-size: 1.5rem;
            margin-bottom: 0.25rem;
        }
        
        .upload-text {
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        
        .upload-filename {
            margin-top: 0.5rem;
            color: var(--success);
            font-weight: 500;
            font-size: 0.85rem;
            word-break: break-all;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            font-size: 0.8rem;
            font-weight: 500;
            margin-bottom: 0.35rem;
            color: var(--text-secondary);
        }
        
        .form-group select,
        .form-group input[type="number"] {
            width: 100%;
            padding: 0.5rem 0.75rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.85rem;
        }
        
        .form-group select:focus,
        .form-group input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        /* Disabled/Coming Soon options */
        .form-group select option.disabled-option {
            color: var(--text-secondary);
            font-style: italic;
        }
        
        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }
        
        .checkbox-item input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: var(--accent);
            cursor: pointer;
        }
        
        .checkbox-item span {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
        }
        
        .btn-primary {
            background: var(--accent);
            color: white;
            width: 100%;
            margin-top: 0.5rem;
        }
        
        .btn-primary:hover:not(:disabled) {
            background: var(--accent-hover);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* History section */
        .history-section {
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }
        
        .history-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .history-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 0.8rem;
        }
        
        .history-item:hover {
            background: var(--bg-tertiary);
        }
        
        .history-item.active {
            background: var(--bg-tertiary);
            border: 1px solid var(--accent);
        }
        
        .history-item .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        
        .history-item .run-info {
            flex: 1;
            overflow: hidden;
        }
        
        .history-item .run-name {
            font-size: 0.75rem;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .history-item .run-time {
            font-size: 0.7rem;
            color: var(--text-secondary);
        }
        
        /* Main content area */
        .main-content {
            flex: 1;
            padding: 1.5rem 2rem;
            overflow-y: auto;
        }
        
        .card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
        }
        
        .card h2 {
            font-size: 1rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-banner {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.875rem 1rem;
            border-radius: 8px;
            background: var(--bg-tertiary);
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        
        .status-indicator.queued { background: var(--text-secondary); }
        .status-indicator.running { background: var(--warning); animation: pulse 1.5s infinite; }
        .status-indicator.completed { background: var(--success); }
        .status-indicator.failed { background: var(--error); }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-info {
            flex: 1;
        }
        
        .status-info .run-id {
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }
        
        .status-info .status-text {
            font-weight: 500;
            text-transform: capitalize;
        }
        
        .elapsed-time {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        /* How it works button */
        .how-it-works-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.4rem 0.75rem;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
            margin-left: 1rem;
        }
        
        .how-it-works-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border-color: var(--accent);
        }
        
        /* Modal Overlay */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s, visibility 0.3s;
        }
        
        .modal-overlay.active {
            opacity: 1;
            visibility: visible;
        }
        
        .modal-content {
            background: var(--bg-secondary);
            border-radius: 12px;
            width: 90%;
            max-width: 900px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            transform: translateY(20px);
            transition: transform 0.3s;
        }
        
        .modal-overlay.active .modal-content {
            transform: translateY(0);
        }
        
        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border);
            flex-shrink: 0;
        }
        
        .modal-header h2 {
            font-size: 1.1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0;
        }
        
        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.25rem;
            line-height: 1;
            transition: color 0.2s;
        }
        
        .modal-close:hover {
            color: var(--text-primary);
        }
        
        .modal-body {
            padding: 1.5rem;
            overflow-y: auto;
            flex: 1;
        }
        
        /* Modal content styling */
        .modal-body h1, .modal-body h2, .modal-body h3 {
            color: var(--text-primary);
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
        
        .modal-body h1:first-child {
            margin-top: 0;
        }
        
        .modal-body h1 {
            font-size: 1.5rem;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
        }
        
        .modal-body h2 {
            font-size: 1.2rem;
            color: var(--accent);
        }
        
        .modal-body h3 {
            font-size: 1rem;
        }
        
        .modal-body p {
            color: var(--text-secondary);
            margin-bottom: 1rem;
            line-height: 1.7;
        }
        
        .modal-body ul, .modal-body ol {
            color: var(--text-secondary);
            padding-left: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .modal-body li {
            margin-bottom: 0.5rem;
        }
        
        .modal-body strong {
            color: var(--text-primary);
        }
        
        .modal-body code {
            background: var(--bg-primary);
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85em;
            color: var(--accent);
        }
        
        .modal-body pre {
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid var(--border);
            margin: 1rem 0;
        }
        
        .modal-body pre code {
            padding: 0;
            background: none;
            color: var(--text-secondary);
            font-size: 0.8rem;
            line-height: 1.5;
        }
        
        .modal-body table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.85rem;
        }
        
        .modal-body th, .modal-body td {
            border: 1px solid var(--border);
            padding: 0.6rem 0.8rem;
            text-align: left;
        }
        
        .modal-body th {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .modal-body tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
        
        .modal-body hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 1.5rem 0;
        }
        
        .modal-body blockquote {
            border-left: 4px solid var(--accent);
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }
        
        .progress-bar {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 0.75rem;
        }
        
        .progress-bar-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s;
        }
        
        .progress-bar-fill.completed { background: var(--success); }
        .progress-bar-fill.failed { background: var(--error); }
        
        .log-output {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 8px;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 0.75rem;
            line-height: 1.5;
            max-height: 350px;
            overflow-y: auto;
            padding: 1rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .log-output:empty::before {
            content: 'Waiting for pipeline to start...';
            color: var(--text-secondary);
        }
        
        .output-files {
            display: grid;
            gap: 0.75rem;
        }
        
        .output-file {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 8px;
            text-decoration: none;
            color: var(--text-primary);
            transition: background 0.2s;
        }
        
        .output-file:hover {
            background: var(--border);
        }
        
        .output-file.primary {
            border: 1px solid var(--accent);
            background: rgba(59, 130, 246, 0.1);
        }
        
        .output-file .file-icon {
            font-size: 1.25rem;
        }
        
        .output-file .file-info {
            flex: 1;
        }
        
        .output-file .file-name {
            font-weight: 500;
            font-size: 0.85rem;
        }
        
        .output-file .file-desc {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .output-file .file-action {
            font-size: 0.8rem;
            color: var(--accent);
        }
        
        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: var(--text-secondary);
        }
        
        .empty-state .icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            border-radius: 8px;
            padding: 1rem;
            color: var(--error);
            font-size: 0.85rem;
            margin-top: 1rem;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="top-header">
        <div class="logo-section">
            <img src="/static/ascendion_logo.png" alt="Ascendion">
            <div>
                <div class="app-title">OCR Pipeline</div>
                <div class="app-subtitle">Document Intelligence Extraction</div>
            </div>
        </div>
    </div>
    
    <div class="main-layout">
        <!-- Left Sidebar - Pinned -->
        <div class="sidebar">
            <h2>⚙️ Configuration</h2>
            
            <form id="uploadForm">
                <div class="upload-area" id="uploadArea">
                    <input type="file" id="pdfInput" name="pdf" accept=".pdf">
                    <div class="upload-icon">📄</div>
                    <div class="upload-text">Drop PDF or click to browse</div>
                    <div class="upload-filename" id="fileName"></div>
                </div>
                
                <div class="form-group">
                    <label for="provider">Provider</label>
                    <select id="provider" name="provider">
                        <option value="azure" selected>Azure Document Intelligence</option>
                        <option value="vertex" disabled class="disabled-option">Google Vertex AI (Coming Soon)</option>
                        <option value="aws" disabled class="disabled-option">AWS Textract (Coming Soon)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="dpi">DPI (150-600)</label>
                    <input type="number" id="dpi" name="dpi" value="300" min="150" max="600" step="50">
                </div>
                
                <div class="form-group">
                    <label>Options</label>
                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" id="enhance" name="enhance" checked>
                            <span>Enhance Images</span>
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" id="save_artifacts" name="save_artifacts">
                            <span>Save Artifacts</span>
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" id="signature_report" name="signature_report" checked>
                            <span>Signature Report</span>
                        </label>
                    </div>
                </div>
                
                <button type="submit" class="btn btn-primary" id="runBtn" disabled>
                    ▶️ Run Pipeline
                </button>
            </form>
            
            <div class="history-section">
                <h2>📜 History</h2>
                <div class="history-list" id="historyList">
                    <div class="empty-state" style="padding: 1rem; font-size: 0.8rem;">
                        No previous runs
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <div class="card">
                <div class="empty-state" id="emptyState">
                    <div class="icon">📋</div>
                    <div>Upload a PDF and run the pipeline to see results</div>
                    <button class="how-it-works-btn" onclick="openHowItWorks()" style="margin-top: 1rem;">
                        ℹ️ How it works
                    </button>
                </div>
                
                <div id="activeRun" style="display: none;">
                    <div class="status-banner">
                        <div class="status-indicator" id="statusIndicator"></div>
                        <div class="status-info">
                            <div class="run-id" id="runIdDisplay"></div>
                            <div class="status-text" id="statusText">Queued</div>
                        </div>
                        <div class="elapsed-time" id="elapsedTime"></div>
                        <button class="how-it-works-btn" onclick="openHowItWorks()">
                            ℹ️ How it works
                        </button>
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress-bar-fill" id="progressBar" style="width: 0%"></div>
                    </div>
                    
                    <div class="error-message" id="errorMessage" style="display: none;"></div>
                </div>
            </div>
            
            <div class="card" id="logCard" style="display: none;">
                <h2>📝 Log Output</h2>
                <div class="log-output" id="logOutput"></div>
            </div>
            
            <div class="card" id="outputsCard" style="display: none;">
                <h2>📦 Output Files</h2>
                <div class="output-files" id="outputFiles"></div>
            </div>
        </div>
    </div>
    
    <!-- How it works Modal -->
    <div class="modal-overlay" id="howItWorksModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>📖 How it works</h2>
                <button class="modal-close" onclick="closeHowItWorks()">×</button>
            </div>
            <div class="modal-body" id="howItWorksContent">
                Loading...
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentRunId = null;
        let pollInterval = null;
        
        // Elements
        const uploadArea = document.getElementById('uploadArea');
        const pdfInput = document.getElementById('pdfInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const runBtn = document.getElementById('runBtn');
        const emptyState = document.getElementById('emptyState');
        const activeRun = document.getElementById('activeRun');
        const statusIndicator = document.getElementById('statusIndicator');
        const runIdDisplay = document.getElementById('runIdDisplay');
        const statusText = document.getElementById('statusText');
        const elapsedTime = document.getElementById('elapsedTime');
        const progressBar = document.getElementById('progressBar');
        const errorMessage = document.getElementById('errorMessage');
        const logCard = document.getElementById('logCard');
        const logOutput = document.getElementById('logOutput');
        const outputsCard = document.getElementById('outputsCard');
        const outputFiles = document.getElementById('outputFiles');
        const historyList = document.getElementById('historyList');
        
        // Upload area interactions
        uploadArea.addEventListener('click', () => pdfInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                pdfInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });
        
        pdfInput.addEventListener('change', handleFileSelect);
        
        function handleFileSelect() {
            const file = pdfInput.files[0];
            if (file) {
                fileName.textContent = file.name;
                uploadArea.classList.add('has-file');
                runBtn.disabled = false;
            } else {
                fileName.textContent = '';
                uploadArea.classList.remove('has-file');
                runBtn.disabled = true;
            }
        }
        
        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('pdf', pdfInput.files[0]);
            formData.append('provider', document.getElementById('provider').value);
            formData.append('dpi', document.getElementById('dpi').value);
            formData.append('enhance', document.getElementById('enhance').checked);
            formData.append('save_artifacts', document.getElementById('save_artifacts').checked);
            formData.append('signature_report', document.getElementById('signature_report').checked);
            
            runBtn.disabled = true;
            runBtn.innerHTML = '⏳ Starting...';
            
            try {
                const response = await fetch('/run', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentRunId = data.run_id;
                    showRunStatus(data.run_id, 'queued');
                    startPolling(data.run_id);
                    loadHistory();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error starting pipeline: ' + error.message);
            } finally {
                runBtn.disabled = false;
                runBtn.innerHTML = '▶️ Run Pipeline';
            }
        });
        
        function showRunStatus(runId, status) {
            emptyState.style.display = 'none';
            activeRun.style.display = 'block';
            logCard.style.display = 'block';
            
            runIdDisplay.textContent = runId;
            updateStatus(status);
        }
        
        function updateStatus(status, elapsed = null, error = null) {
            statusIndicator.className = 'status-indicator ' + status;
            statusText.textContent = status;
            
            if (elapsed !== null) {
                const mins = Math.floor(elapsed / 60);
                const secs = Math.floor(elapsed % 60);
                elapsedTime.textContent = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
            }
            
            // Progress bar
            progressBar.className = 'progress-bar-fill';
            if (status === 'queued') {
                progressBar.style.width = '5%';
            } else if (status === 'running') {
                progressBar.style.width = '50%';
            } else if (status === 'completed') {
                progressBar.style.width = '100%';
                progressBar.classList.add('completed');
            } else if (status === 'failed') {
                progressBar.style.width = '100%';
                progressBar.classList.add('failed');
            }
            
            // Error message
            if (error) {
                errorMessage.textContent = error;
                errorMessage.style.display = 'block';
            } else {
                errorMessage.style.display = 'none';
            }
        }
        
        function startPolling(runId) {
            if (pollInterval) {
                clearInterval(pollInterval);
            }
            
            async function poll() {
                try {
                    const response = await fetch(`/status/${runId}`);
                    const data = await response.json();
                    
                    updateStatus(data.status, data.elapsed_time, data.error);
                    
                    // Update log
                    if (data.log) {
                        logOutput.textContent = data.log;
                        logOutput.scrollTop = logOutput.scrollHeight;
                    }
                    
                    // If completed or failed, load outputs and stop polling
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(pollInterval);
                        pollInterval = null;
                        if (data.status === 'completed') {
                            loadOutputs(runId);
                        }
                        loadHistory();
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                }
            }
            
            poll();
            pollInterval = setInterval(poll, 2000);
        }
        
        async function loadOutputs(runId) {
            try {
                const response = await fetch(`/outputs/${runId}`);
                const data = await response.json();
                
                if (data.files && data.files.length > 0) {
                    outputsCard.style.display = 'block';
                    outputFiles.innerHTML = data.files.map((file, idx) => {
                        const icon = getFileIcon(file.type, file.category);
                        const desc = getFileDescription(file);
                        const isPrimary = file.category === 'document' || file.category === 'signature';
                        return `
                            <a href="${file.view_url}" target="_blank" class="output-file ${isPrimary ? 'primary' : ''}">
                                <span class="file-icon">${icon}</span>
                                <div class="file-info">
                                    <div class="file-name">${file.name}</div>
                                    <div class="file-desc">${desc}</div>
                                </div>
                                <span class="file-action">View →</span>
                            </a>
                        `;
                    }).join('');
                }
            } catch (error) {
                console.error('Error loading outputs:', error);
            }
        }
        
        function getFileIcon(type, category) {
            if (category === 'document') return '📄';
            if (category === 'signature') return '✍️';
            const icons = {
                markdown: '📝',
                json: '📋',
                log: '📜',
                image: '🖼️',
                file: '📎'
            };
            return icons[type] || '📎';
        }
        
        function getFileDescription(file) {
            if (file.category === 'document') return 'Extracted document with tables';
            if (file.category === 'signature') return 'Signature analysis report';
            if (file.type === 'json') return 'Structured data';
            if (file.type === 'log') return 'Processing log';
            if (file.type === 'image') return 'Signature image';
            return formatSize(file.size);
        }
        
        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        async function loadHistory() {
            try {
                const response = await fetch('/runs');
                const data = await response.json();
                
                if (data.runs && data.runs.length > 0) {
                    historyList.innerHTML = data.runs.map(run => {
                        const isActive = run.run_id === currentRunId;
                        const time = new Date(run.created_at).toLocaleString();
                        return `
                            <div class="history-item ${isActive ? 'active' : ''}" onclick="selectRun('${run.run_id}')">
                                <div class="status-dot status-indicator ${run.status}"></div>
                                <div class="run-info">
                                    <div class="run-name">${run.run_id}</div>
                                    <div class="run-time">${time}</div>
                                </div>
                            </div>
                        `;
                    }).join('');
                }
            } catch (error) {
                console.error('Error loading history:', error);
            }
        }
        
        function selectRun(runId) {
            currentRunId = runId;
            showRunStatus(runId, 'loading');
            outputsCard.style.display = 'none';
            
            // Stop any existing polling
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
            
            // Fetch current status
            fetch(`/status/${runId}`)
                .then(r => r.json())
                .then(data => {
                    updateStatus(data.status, data.elapsed_time, data.error);
                    if (data.log) {
                        logOutput.textContent = data.log;
                        logOutput.scrollTop = logOutput.scrollHeight;
                    }
                    
                    if (data.status === 'running' || data.status === 'queued') {
                        startPolling(runId);
                    } else if (data.status === 'completed') {
                        loadOutputs(runId);
                    }
                    
                    loadHistory();
                });
        }
        
        // Initial load
        loadHistory();
        
        // How it works modal
        const howItWorksModal = document.getElementById('howItWorksModal');
        const howItWorksContent = document.getElementById('howItWorksContent');
        let howItWorksLoaded = false;
        
        async function openHowItWorks() {
            howItWorksModal.classList.add('active');
            document.body.style.overflow = 'hidden';
            
            if (!howItWorksLoaded) {
                try {
                    const response = await fetch('/how-it-works');
                    const data = await response.json();
                    howItWorksContent.innerHTML = data.html;
                    howItWorksLoaded = true;
                } catch (error) {
                    howItWorksContent.innerHTML = '<p>Failed to load content. Please try again.</p>';
                }
            }
        }
        
        function closeHowItWorks() {
            howItWorksModal.classList.remove('active');
            document.body.style.overflow = '';
        }
        
        // Close modal on overlay click
        howItWorksModal.addEventListener('click', (e) => {
            if (e.target === howItWorksModal) {
                closeHowItWorks();
            }
        });
        
        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && howItWorksModal.classList.contains('active')) {
                closeHowItWorks();
            }
        });
    </script>
</body>
</html>
"""

# Document template - for extracted markdown with proper table formatting
DOCUMENT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        :root {
            --bg: #ffffff;
            --text: #1a1a1a;
            --text-secondary: #666666;
            --border: #e0e0e0;
            --header-bg: #f8f9fa;
            --accent: #2563eb;
        }
        
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #1a1a2e;
                --text: #e8e8e8;
                --text-secondary: #a0a0a0;
                --border: #3a3a4a;
                --header-bg: #252538;
                --accent: #60a5fa;
            }
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg);
            color: var(--text);
            line-height: 1.8;
            font-size: 16px;
        }
        
        /* Header bar with download */
        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            margin-bottom: 2rem;
            background: var(--header-bg);
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        .doc-title {
            font-size: 1rem;
            color: var(--text-secondary);
        }
        
        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--accent);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .download-btn:hover {
            opacity: 0.9;
        }
        
        /* Document content styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text);
            margin-top: 2rem;
            margin-bottom: 1rem;
            line-height: 1.3;
        }
        
        h1 {
            font-size: 1.75rem;
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
        }
        
        h2 {
            font-size: 1.4rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.3rem;
        }
        
        h3 { font-size: 1.2rem; }
        h4 { font-size: 1.1rem; }
        
        p {
            margin-bottom: 1rem;
            text-align: justify;
        }
        
        /* Table styling - document-like appearance */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 0.9rem;
        }
        
        th, td {
            border: 1px solid var(--border);
            padding: 0.75rem 1rem;
            text-align: left;
            vertical-align: top;
        }
        
        th {
            background: var(--header-bg);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }
        
        tr:nth-child(even) {
            background: rgba(0, 0, 0, 0.02);
        }
        
        @media (prefers-color-scheme: dark) {
            tr:nth-child(even) {
                background: rgba(255, 255, 255, 0.02);
            }
        }
        
        /* Numeric cells - right align */
        td:last-child {
            text-align: right;
        }
        
        /* Code blocks */
        code {
            background: var(--header-bg);
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9em;
        }
        
        pre {
            background: var(--header-bg);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid var(--border);
        }
        
        pre code {
            padding: 0;
            background: none;
        }
        
        /* Lists */
        ul, ol {
            margin-bottom: 1rem;
            padding-left: 2rem;
        }
        
        li {
            margin-bottom: 0.5rem;
        }
        
        /* Blockquotes */
        blockquote {
            border-left: 4px solid var(--accent);
            margin: 1.5rem 0;
            padding: 0.5rem 1rem;
            background: var(--header-bg);
            font-style: italic;
        }
        
        /* Images */
        img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        /* Horizontal rule */
        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 2rem 0;
        }
        
        /* Print styles */
        @media print {
            body {
                max-width: none;
                padding: 0;
            }
            .doc-header {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="doc-header">
        <div class="doc-title">📄 {{title}}</div>
        {{download_link}}
    </div>
    <div class="doc-content">
        {{content}}
    </div>
</body>
</html>
"""

# Signature report template - with embedded images
SIGNATURE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        :root {
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #e2e8f0;
            --text-secondary: #94a3b8;
            --border: #334155;
            --accent: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }
        
        /* Header */
        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            margin-bottom: 2rem;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        
        .report-title {
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--accent);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.85rem;
        }
        
        .download-btn:hover {
            opacity: 0.9;
        }
        
        /* Content styling */
        h1, h2, h3, h4 {
            color: var(--text);
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h1 {
            font-size: 1.5rem;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
        }
        
        h2 {
            font-size: 1.25rem;
            color: var(--accent);
        }
        
        h3 {
            font-size: 1.1rem;
        }
        
        p {
            margin-bottom: 1rem;
            color: var(--text-secondary);
        }
        
        strong {
            color: var(--text);
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
        }
        
        th, td {
            border: 1px solid var(--border);
            padding: 0.75rem 1rem;
            text-align: left;
        }
        
        th {
            background: rgba(59, 130, 246, 0.2);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
        
        /* Signature images */
        img {
            max-width: 200px;
            height: auto;
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 0.5rem;
            background: white;
            margin: 0.5rem 0;
        }
        
        /* Lists */
        ul, ol {
            margin-bottom: 1rem;
            padding-left: 1.5rem;
            color: var(--text-secondary);
        }
        
        li {
            margin-bottom: 0.5rem;
        }
        
        /* Code */
        code {
            background: var(--bg);
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.9em;
            color: var(--accent);
        }
        
        pre {
            background: var(--bg);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid var(--border);
        }
        
        pre code {
            padding: 0;
            background: none;
        }
        
        /* Blockquotes */
        blockquote {
            border-left: 4px solid var(--accent);
            margin: 1.5rem 0;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            color: var(--text-secondary);
        }
        
        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 2rem 0;
        }
        
        /* Status badges in content */
        .status-consistent { color: var(--success); }
        .status-discrepancy { color: var(--error); }
        .status-warning { color: var(--warning); }
    </style>
</head>
<body>
    <div class="report-header">
        <div class="report-title">✍️ Signature Analysis Report</div>
        {{download_link}}
    </div>
    <div class="report-content">
        {{content}}
    </div>
</body>
</html>
"""

MARKDOWN_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.7;
        }
        h1, h2, h3, h4 { color: #f1f5f9; margin-top: 1.5em; }
        h1 { border-bottom: 2px solid #3b82f6; padding-bottom: 0.5rem; }
        a { color: #60a5fa; }
        code {
            background: #1e293b;
            padding: 0.2em 0.4em;
            border-radius: 4px;
            font-family: 'SF Mono', Monaco, monospace;
        }
        pre {
            background: #1e293b;
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
        }
        pre code { padding: 0; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        th, td {
            border: 1px solid #475569;
            padding: 0.75rem;
            text-align: left;
        }
        th { background: #1e293b; }
        tr:nth-child(even) { background: rgba(30, 41, 59, 0.5); }
        blockquote {
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
            padding-left: 1rem;
            color: #94a3b8;
        }
        img { max-width: 100%; height: auto; border-radius: 8px; }
        .download-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: #3b82f6;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body>
{{download_link}}
{{content}}
</body>
</html>
"""

JSON_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        body {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem;
            margin: 0;
        }
        pre {
            background: #1e293b;
            padding: 1.5rem;
            border-radius: 8px;
            overflow-x: auto;
            line-height: 1.5;
            font-size: 0.9rem;
        }
        .string { color: #86efac; }
        .number { color: #fbbf24; }
        .boolean { color: #f472b6; }
        .null { color: #94a3b8; }
        .key { color: #60a5fa; }
    </style>
</head>
<body>
<pre id="json">{{content}}</pre>
<script>
    const json = document.getElementById('json');
    const text = json.textContent;
    const highlighted = text.replace(
        /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        (match) => {
            let cls = 'number';
            if (/^"/.test(match)) {
                cls = /:$/.test(match) ? 'key' : 'string';
            } else if (/true|false/.test(match)) {
                cls = 'boolean';
            } else if (/null/.test(match)) {
                cls = 'null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        }
    );
    json.innerHTML = highlighted;
</script>
</body>
</html>
"""

LOG_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        body {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem;
            margin: 0;
        }
        pre {
            background: #1e293b;
            padding: 1.5rem;
            border-radius: 8px;
            overflow-x: auto;
            line-height: 1.5;
            font-size: 0.85rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
    </style>
</head>
<body>
<pre>{{content}}</pre>
</body>
</html>
"""


# ============================================================================
# Main Entry
# ============================================================================

if __name__ == "__main__":
    print(f"Starting OCR Pipeline UI on http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
