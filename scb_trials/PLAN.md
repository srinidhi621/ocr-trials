# Plan to Close Gaps (Azure DI Only)

## Goals
- Improve signature detection recall and validation without missing signatures.
- Preserve table structure and document layout faithfully.
- Add measurable accuracy checks to prevent regressions.

## Scope Assumptions
- Azure Document Intelligence is always available.
- Output is always Markdown + JSON.
- Credentials are stored locally and never committed.

## Workstreams

### 1) Signature Detection & Consistency (Top Priority) [DONE]
- Goal: near‑zero missed signatures and reliable consistency checks.
- Detection upgrades:
  - Dedicated signature scan pass that combines DI + CV + LLM validation.
  - Multi‑region scanning (full page + footer + margins).
  - Adaptive thresholds by page type (dense/blank pages).
  - De‑duplication and clustering across overlapping candidates.
- Consistency upgrades:
  - Normalize signer identity (name + designation + visual cues).
  - Compare within clusters only; report confidence bands.
  - Flag low‑confidence comparisons for manual review.
- Coverage metrics:
  - “Signature coverage” per page (candidates vs. confirmed).
  - “Signer stability” score per signer group.

### 2) Local UI (Lightweight) [NEEDS REBUILD]
Single-page Flask UI for running the OCR pipeline locally.

#### Functional Requirements
1. **PDF Upload**: File input to select a PDF document
2. **Configuration Options**:
   - Provider: dropdown (azure, vertex)
   - DPI: number input (default 300, range 150-600)
   - Enhance: checkbox (default checked)
   - Save Artifacts: checkbox
   - Signature Report: checkbox (default checked)
3. **Run Pipeline**: Button that triggers `main.run_pipeline()` in a background thread
4. **Progress Display**:
   - Run ID (generated from filename + provider + timestamp)
   - Status indicator (queued → running → completed/failed)
   - Log output area (tail of `{run_id}.log` file, auto-scroll)
   - Progress bar with elapsed time and ETA estimate
5. **Output Links** (after completion):
   - `*_extracted.md` - rendered as HTML
   - `*_extracted.json` - syntax highlighted
   - `*_signature_report.md` - rendered as HTML
   - `signatures/` directory listing

#### Technical Requirements
- Flask backend on port 5001 (configurable via PORT env var)
- Pipeline runs in daemon thread (non-blocking)
- Status polling via `/status/<run_id>` endpoint (every 2s)
- Outputs served via `/view/<run_id>/<path>` with proper rendering
- Support page refresh / browser private mode (persist run state to disk, not just memory)
- Scripts: `scripts/ui_up.sh` and `scripts/ui_down.sh` for start/stop

#### API Endpoints
- `GET /` - Serve the HTML page
- `POST /run` - Upload PDF, start pipeline, return JSON `{run_id, status, output_dir}`
- `GET /status/<run_id>` - Return JSON `{run_id, status, log, output_dir, error}`
- `GET /outputs/<run_id>` - Return JSON list of output files with view URLs
- `GET /view/<run_id>/<path>` - Render markdown/JSON or serve file
- `GET /files/<run_id>/<path>` - Serve raw file or directory listing

### 3) Final Review Pass (Vision LLM)
- Add optional “final review” pass that:
  - Sends the original PDF pages and extracted outputs to a vision‑enabled LLM.
  - Requests a coverage checklist (missing signatures, tables, redactions, key values).
  - Produces a structured gap report with page references.
- If gaps are detected:
  - Re‑run the specific stage (signature/table/redaction) for the affected pages.
  - Append the corrected outputs and update the confidence report.

### 4) Table and Structure Fidelity (High Impact)
- Enforce DI table structure as the source of truth.
- Validate LLM‑refined table markdown dimensions; fallback to DI.
- Add a layout‑driven reading order pass that:
  - Uses region coordinates for ordering.
  - Groups headers/footers separately.
- Add table integrity checks:
  - Row/column count match.
  - Empty cell count thresholds.

### 5) Golden Tests & Regression Harness
- Build a small golden dataset (3–5 PDFs).
- Store expected JSON/Markdown outputs for diffing.
- Add automated diff checks with tolerance rules (timestamps, IDs).
- Add a CI smoke test for the pipeline using cached artifacts.

### 6) Financial Extraction Coverage
- Expand regex patterns:
  - Negative values, parentheses, Indian separators, “/-”.
  - Abbreviated units (bn/mn/cr) and currency suffixes.
- Add numeric normalization tests for common formats.

### 7) Redaction Detection Resilience
- Detect light/blurred redactions using threshold ranges.
- Add a fallback to detect “solid blocks” via low texture/variance.

### 8) Signature Report Consistency
- Align signature report comparison IDs with signature snippet IDs.
- Ensure detailed comparisons map to the correct signature pairs.
- Include coverage + consistency metrics from workstream #1.


## Deliverables
- Updated pipeline modules with improved detection and fidelity.
- Golden tests + regression harness.
- Documented metrics in output JSON (coverage, integrity flags).
- Local UI for upload, progress, and report links.

## Timeline (Suggested)
- Phase 1 (2–4 days): Signature detection + consistency upgrades (priority).
- Phase 2 (1–2 days): Local UI + final review pass.
- Phase 3 (1–2 days): Table fidelity updates.
- Phase 4 (2–3 days): Golden tests + financial coverage.
- Phase 5 (1–2 days): Redaction + signature report consistency.
