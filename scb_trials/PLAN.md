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

### 1) Signature Detection Robustness (High Impact)
- Add a dedicated signature scan pass that:
  - Uses DI layout for candidate regions.
  - Runs a CV pass over full page to catch misses.
  - Uses LLM validation on all candidates and rejects non‑signatures.
- Add recall-oriented thresholds and configurable search regions.
- Record “signature coverage” per page (detected vs. candidate count).
- Add confidence flags for any page with low signature confidence.

### 2) Table and Structure Fidelity (High Impact)
- Enforce DI table structure as the source of truth.
- Validate LLM‑refined table markdown dimensions; fallback to DI.
- Add a layout‑driven reading order pass that:
  - Uses region coordinates for ordering.
  - Groups headers/footers separately.
- Add table integrity checks:
  - Row/column count match.
  - Empty cell count thresholds.

### 3) Golden Tests & Regression Harness
- Build a small golden dataset (3–5 PDFs).
- Store expected JSON/Markdown outputs for diffing.
- Add automated diff checks with tolerance rules (timestamps, IDs).
- Add a CI smoke test for the pipeline using cached artifacts.

### 4) Financial Extraction Coverage
- Expand regex patterns:
  - Negative values, parentheses, Indian separators, “/-”.
  - Abbreviated units (bn/mn/cr) and currency suffixes.
- Add numeric normalization tests for common formats.

### 5) Redaction Detection Resilience
- Detect light/blurred redactions using threshold ranges.
- Add a fallback to detect “solid blocks” via low texture/variance.

### 6) Signature Report Consistency
- Align signature report comparison IDs with signature snippet IDs.
- Ensure detailed comparisons map to the correct signature pairs.

## Deliverables
- Updated pipeline modules with improved detection and fidelity.
- Golden tests + regression harness.
- Documented metrics in output JSON (coverage, integrity flags).

## Timeline (Suggested)
- Phase 1 (1–2 days): Signature + table fidelity updates.
- Phase 2 (2–3 days): Golden tests + financial coverage.
- Phase 3 (1–2 days): Redaction + signature report consistency.
