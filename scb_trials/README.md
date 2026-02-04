# Document OCR Pipeline

A comprehensive 4-stage document extraction pipeline for scanned PDFs using Azure Document Intelligence and multimodal LLMs (GPT-5.2 / Gemini 3 Pro).

## Status (February 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| **Azure Provider** | ✅ Production Ready | Fully working with Document Intelligence + GPT-5.2 |
| **Signature Detection** | ✅ Working | Hybrid CV + LLM detection |
| **Signature Comparison** | ✅ Working | Multi-metric similarity scoring (SSIM + ORB + pHash) |
| **Signature Reports** | ✅ Working | Dedicated JSON/Markdown reports |
| **Run ID & Logging** | ✅ Working | Per-run directories with full debug logs |
| **Vertex Provider** | ⚠️ Needs Work | Slow API responses, not recommended |

### Latest Test Results (11-page SCB Facility Letter)

| Metric | Value |
|--------|-------|
| Processing Time | ~9.5 minutes |
| Text Blocks | 428 |
| Tables Extracted | 13 |
| Signatures Detected | 18 |
| Financial Values | 77 |
| Redactions | 3 |
| Overall Confidence | 88.79% |

## Purpose

Extract structured content from scanned PDF documents with high accuracy, specifically designed for:

- **Table Extraction**: Exact structure preservation in Markdown format
- **Financial Precision**: Verbatim extraction of monetary values (no unit conversion)
- **Signature Analysis**: Detection, extraction, visual comparison, and matching across pages
- **Redaction Detection**: Identification of blacked-out regions marked as `[REDACTED]`

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the pipeline (basic)
python main.py document.pdf --provider azure

# 3. Run with all features (recommended)
python main.py document.pdf --provider azure --signature-report --save-artifacts -v
```

## Architecture

The pipeline consists of four sequential stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI INTERFACE                                │
│     python main.py <pdf> --provider azure --signature-report         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: PRE-PROCESSING                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  PDF → Images   │  │ Image Enhance   │  │  Page Preparation   │  │
│  │  (pdf2image)    │  │ (OpenCV)        │  │  (deskew, denoise)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: OCR / EXTRACTION                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Azure Document Intelligence + GPT-5.2                        │   │
│  │  (layout, tables, text, signatures, redactions)               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              +                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Hybrid Signature Detection (CV contours + LLM validation)    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: POST-PROCESSING                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ Content Merger  │  │   Signature     │  │    Financial        │  │
│  │ (reading order) │  │   Comparator    │  │    Validator        │  │
│  └─────────────────┘  │ (SSIM+ORB+pHash)│  └─────────────────────┘  │
│                       └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: REVIEW & OUTPUT                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   Confidence    │  │   Extraction    │  │    Signature        │  │
│  │   Scoring       │  │   Reports       │  │    Reports          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
scb_trials/
├── main.py                        # CLI entry point with run ID & logging
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py            # Stage 1: PDF conversion, enhancement
│   ├── extractor.py               # Stage 2: OCR + hybrid signature detection
│   ├── postprocessor.py           # Stage 3: Merging, signature comparison
│   ├── reviewer.py                # Stage 4: Confidence scoring, output
│   ├── signature_analyzer.py      # Signature grouping & analysis
│   └── signature_report.py        # Signature report generator
├── providers/
│   ├── base.py                    # Abstract provider interface
│   ├── azure_provider.py          # Azure DI + GPT-5.2
│   └── vertex_provider.py         # Google Doc AI + Gemini 3 Pro
├── utils/
│   ├── image_utils.py             # OpenCV operations
│   └── financial_utils.py         # Financial value extraction
├── prompts/                       # LLM prompt templates
├── output/                        # Generated outputs (per run_id)
├── .env                           # Environment credentials
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- Poppler (for PDF to image conversion)

### Setup

1. **Install Poppler** (macOS):
   ```bash
   brew install poppler
   ```

2. **Create virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure credentials** - Create a `.env` file:
   ```env
   # Azure OpenAI
   AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.2

   # Azure Document Intelligence
   AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key
   ```

## Usage

### CLI Options

```bash
python main.py <pdf_path> [OPTIONS]

Options:
  -p, --provider [azure|vertex]       OCR provider (required)
  --dpi INTEGER                       DPI for conversion (default: 300)
  --enhance / --no-enhance            Image enhancement (default: enabled)
  -o, --output-dir PATH               Output directory (default: ./output)
  --save-artifacts / --no-artifacts   Save intermediate images
  --signature-report / --no-signature-report  Generate signature report (default: enabled)
  -v, --verbose                       Verbose output
  --help                              Show help
```

### Examples

```bash
# Basic extraction
python main.py document.pdf --provider azure

# Full extraction with all features
python main.py document.pdf --provider azure --signature-report --save-artifacts -v

# High DPI for better quality
python main.py document.pdf --provider azure --dpi 400 --save-artifacts -v

# Custom output directory
python main.py document.pdf --provider azure -o ./my_output -v
```

### Run ID System

Each run generates a unique **Run ID** in the format:
```
{document_name}_{provider}_{YYYYMMDD_HHMMSS}
```

Example: `supplementary_facility_letter_azure_20260204_164645`

All outputs are saved in a dedicated directory under `output/{run_id}/`.

## Output Structure

After running the pipeline, outputs are organized per run:

```
output/
└── supplementary_facility_letter_azure_20260204_164645/
    ├── artifacts/                                    # Page images
    │   ├── page_001_original.png
    │   ├── page_001_processed.png
    │   └── ...
    ├── signatures/                                   # Extracted signature snippets
    │   ├── sig_page006_00.png
    │   └── sig_page011_01.png
    ├── supplementary_facility_letter_azure_20260204_164645.log  # Full debug log
    ├── supplementary_facility_letter_extracted.md               # Main report (Markdown)
    ├── supplementary_facility_letter_extracted.json             # Main report (JSON)
    ├── supplementary_facility_letter_confidence.json            # Confidence metrics
    ├── supplementary_facility_letter_signature_report.json      # Signature analysis (JSON)
    └── supplementary_facility_letter_signature_report.md        # Signature analysis (Markdown)
```

### Output Files Explained

| File | Description |
|------|-------------|
| `*_extracted.md` | Human-readable extraction report with tables, text, signatures |
| `*_extracted.json` | Machine-readable extraction data |
| `*_confidence.json` | Quality metrics, warnings, review flags |
| `*_signature_report.json` | Detailed signature analysis with comparisons |
| `*_signature_report.md` | Human-readable signature report |
| `*.log` | Complete debug log with timestamps |
| `artifacts/` | Original and processed page images |
| `signatures/` | Cropped signature image snippets |

## Viewing Results

### 1. Quick Summary (Console)

After running, the console displays a summary table:

```
                         Extraction Summary                             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Value                                           ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Run ID             │ document_azure_20260204_164645                  │
│ Total Pages        │ 11                                              │
│ Tables Extracted   │ 13                                              │
│ Signatures Found   │ 18                                              │
│ Financial Values   │ 77                                              │
│ Overall Confidence │ 88.8%                                           │
│ Processing Time    │ 574.93s                                         │
└────────────────────┴─────────────────────────────────────────────────┘
```

### 2. Signature Analysis Report

Open `*_signature_report.md` to see:

```markdown
# Signature Analysis Report

**Source:** document.pdf | **Date:** 2026-02-04
**Signatures Detected:** 18 | **Unique Signers:** 2

## Summary by Signer

| Signer | Designation | Occurrences | Pages | Consistency | Avg Score |
|--------|-------------|-------------|-------|-------------|-----------|
| [MANAGING DIRECTOR] | MANAGING DIRECTOR | 9 | 1,2,3,4,5,7,8,9,10 | Discrepancy | 0.28 |
| Unknown Signer | - | 9 | 2,3,4,6,7,8,9,10,11 | Inconsistent | 0.17 |

## Detailed Analysis: [MANAGING DIRECTOR]

### Signature Instances
| ID | Page | Confidence | Visual Description |
|----|------|------------|-------------------|
| sig_p01_000 | 1 | 80% | Handwritten cursive-style signature... |
...
```

### 3. Debug Logs

For troubleshooting, check the `.log` file:

```
2026-02-04 16:46:45 | INFO     | ocr_pipeline | ============================================================
2026-02-04 16:46:45 | INFO     | ocr_pipeline | OCR Pipeline Run Started
2026-02-04 16:46:45 | INFO     | ocr_pipeline | Run ID: supplementary_facility_letter_azure_20260204_164645
...
2026-02-04 16:56:20 | INFO     | ocr_pipeline |   - Signatures detected: 18
2026-02-04 16:56:20 | DEBUG    | ocr_pipeline |   Page 1: 51 text, 1 tables, 1 sigs, confidence=83.13%
...
```

### 4. JSON Data (Programmatic Access)

Parse `*_extracted.json` for programmatic access:

```python
import json

with open('output/run_id/document_extracted.json') as f:
    data = json.load(f)
    
print(f"Pages: {data['metadata']['pages']}")
print(f"Signatures: {len(data['signatures'])}")
print(f"Financial values: {len(data['financial_values'])}")
```

## Signature Analysis

### Detection Methods

The pipeline uses **hybrid signature detection**:

1. **CV-based Detection** (OpenCV)
   - Contour analysis with aspect ratio filtering (1.5:1 to 12:1)
   - Ink density filtering (2-60%)
   - Location heuristics (lower half of page)

2. **LLM-based Detection** (GPT-5.2)
   - Validates CV detections
   - Extracts name, designation, date, visual description
   - Provides semantic understanding

### Comparison Metrics

Signatures are compared using three metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| SSIM | 40% | Structural Similarity Index |
| ORB | 35% | Feature matching (keypoints) |
| pHash | 25% | Perceptual hash distance |

### Comparison Verdicts

| Weighted Score | Verdict |
|----------------|---------|
| ≥ 0.70 | **Match** |
| ≥ 0.50 | **Possible Match** |
| < 0.50 | **No Match** |

## Performance

Tested on an 11-page Standard Chartered Bank Supplemental Facility Letter:

| Metric | Azure + Doc Intelligence |
|--------|--------------------------|
| Processing Time | ~9.5 minutes |
| Text Blocks | 428 |
| Tables Extracted | 13 |
| Signatures Detected | 18 |
| Financial Values | 77 |
| Redactions | 3 |
| Overall Confidence | 88.79% |

## Fallback Mode

If Azure Document Intelligence credentials are not configured, the pipeline automatically falls back to **LLM-only mode**, using GPT-5.2's multimodal capabilities directly for layout analysis.

## Dependencies

```
# Core
openai>=1.0.0
python-dotenv>=1.0.0

# Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-image>=0.22.0
numpy>=1.24.0

# PDF Processing
pdf2image>=1.16.0
PyMuPDF>=1.23.0

# Azure Services
azure-ai-formrecognizer>=3.3.0
azure-identity>=1.15.0

# CLI
click>=8.1.0
rich>=13.0.0
```

## Known Issues

### Signature Similarity Scores

Current signature comparison may show "Discrepancy" even for valid signatures due to:
- Different signing angles/pressure across pages
- Variations in scan quality
- The same title (e.g., "MANAGING DIRECTOR") may have different signers

**Recommendation:** Review the visual descriptions in the signature report to manually verify matches.

### Vertex/Gemini Provider

The Gemini pipeline is currently not production-ready due to slow API response times. Use the Azure provider for all production workloads.

## License

Internal use only.
