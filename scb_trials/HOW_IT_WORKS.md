# Document OCR Pipeline

A comprehensive 4-stage document extraction pipeline for scanned PDFs using Azure Document Intelligence and multimodal LLMs (GPT-5.2 / Gemini 3 Pro).

---

## Purpose

Extract structured content from scanned PDF documents with high accuracy, specifically designed for:

- **Table Extraction**: Exact structure preservation in Markdown format
- **Financial Precision**: Verbatim extraction of monetary values (no unit conversion)
- **Signature Analysis**: Detection, extraction, visual comparison, and matching across pages
- **Redaction Detection**: Identification of blacked-out regions marked as `[REDACTED]`

---

## Architecture

The pipeline consists of four sequential stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                    │
│                                                                              │
│   CLI: python main.py <pdf> --provider azure [options]                       │
│   UI:  python app.py  →  http://localhost:5001                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 1: PRE-PROCESSING                                │
│                                                                              │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│   │   PDF → Images   │ → │  Image Enhance   │ → │ Page Preparation │        │
│   │   (pdf2image)    │   │    (OpenCV)      │   │ (deskew/denoise) │        │
│   └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│                                                                              │
│   Output: List of processed page images (PIL Images)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 2: OCR / EXTRACTION                              │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │         Azure Document Intelligence (Layout Analysis)               │    │
│   │         • Text blocks with bounding boxes                           │    │
│   │         • Table structure extraction                                │    │
│   │         • Reading order detection                                   │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                      +                                       │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │         Hybrid Signature Detection                                  │    │
│   │         • CV contour analysis (aspect ratio, ink density)           │    │
│   │         • LLM validation (GPT-5.2) for semantic understanding       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                      +                                       │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │         LLM Enhancement (GPT-5.2 / Gemini 3 Pro)                    │    │
│   │         • Table markdown formatting                                 │    │
│   │         • Signature metadata extraction                             │    │
│   │         • Financial value identification                            │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Output: ExtractionResult (pages, tables, signatures, text_blocks)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 3: POST-PROCESSING                               │
│                                                                              │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│   │  Content Merger  │   │    Signature     │   │    Financial     │        │
│   │ (reading order)  │   │   Comparator     │   │    Validator     │        │
│   │                  │   │ (SSIM+ORB+pHash) │   │                  │        │
│   └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│                                                                              │
│   Signature Comparison Metrics:                                              │
│   • SSIM (40%) - Structural Similarity Index                                 │
│   • ORB  (35%) - Feature matching (keypoints)                                │
│   • pHash(25%) - Perceptual hash distance                                    │
│                                                                              │
│   Output: ProcessedDocument with merged content and signature analysis       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 4: REVIEW & OUTPUT                               │
│                                                                              │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│   │   Confidence     │   │   Extraction     │   │    Signature     │        │
│   │    Scoring       │   │    Reports       │   │    Reports       │        │
│   │                  │   │  (.md + .json)   │   │  (.md + .json)   │        │
│   └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│                                                                              │
│   Output Files:                                                              │
│   • *_extracted.md      - Human-readable extraction report                   │
│   • *_extracted.json    - Machine-readable extraction data                   │
│   • *_confidence.json   - Quality metrics and warnings                       │
│   • *_signature_report  - Detailed signature analysis (.md + .json)          │
│   • *.log               - Complete debug log                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
scb_trials/
├── main.py                        # CLI entry point with run ID & logging
├── app.py                         # Flask web UI (single-page application)
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py            # Stage 1: PDF conversion, image enhancement
│   ├── extractor.py               # Stage 2: OCR + hybrid signature detection
│   ├── postprocessor.py           # Stage 3: Content merging, signature comparison
│   ├── reviewer.py                # Stage 4: Confidence scoring, output generation
│   ├── signature_analyzer.py      # Signature grouping & analysis logic
│   └── signature_report.py        # Signature report generator
├── providers/
│   ├── __init__.py
│   ├── base.py                    # Abstract provider interface
│   ├── azure_provider.py          # Azure Document Intelligence + GPT-5.2
│   └── vertex_provider.py         # Google Document AI + Gemini 3 Pro
├── utils/
│   ├── __init__.py
│   ├── image_utils.py             # OpenCV operations (deskew, enhance)
│   └── financial_utils.py         # Financial value extraction utilities
├── prompts/
│   ├── table_extraction.txt       # LLM prompt for table formatting
│   ├── text_extraction.txt        # LLM prompt for text analysis
│   └── signature_analysis.txt     # LLM prompt for signature validation
├── scripts/
│   ├── ui_up.sh                   # Start the web UI server
│   └── ui_down.sh                 # Stop the web UI server
├── static/
│   └── ascendion_logo.png         # UI logo asset
├── inputs/                        # Sample PDF documents for testing
├── uploads/                       # UI uploaded files (temporary)
├── output/                        # Generated outputs (per run_id)
├── .env                           # Environment credentials (not in repo)
├── requirements.txt               # Python dependencies
├── HOW_IT_WORKS.md                # This documentation file
└── README.md                      # Project README
```

---

## Usage

### Web UI (Recommended)

The web UI provides an easy-to-use interface for document processing:

```bash
# Start the UI server
./scripts/ui_up.sh
# Or directly:
python app.py

# Access at http://localhost:5001
```

**Features:**
- Drag-and-drop PDF upload
- Real-time progress tracking with live logs
- Configurable options (DPI, provider, artifacts)
- Output file viewer with formatted rendering
- Run history for quick access to past extractions

### Command Line Interface (CLI)

For scripting and automation:

```bash
# Basic extraction
python main.py document.pdf --provider azure

# Full extraction with all features
python main.py document.pdf --provider azure --signature-report --save-artifacts -v

# High DPI for better quality
python main.py document.pdf --provider azure --dpi 400 --save-artifacts -v

# Custom output directory
python main.py document.pdf --provider azure -o ./my_output -v

# Parallel processing control
python main.py document.pdf --provider azure --parallel --max-workers 5
```

**CLI Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--provider`, `-p` | OCR provider (`azure` or `vertex`) | Required |
| `--dpi` | DPI for PDF conversion (150-600) | 300 |
| `--enhance/--no-enhance` | Image enhancement | Enabled |
| `--output-dir`, `-o` | Output directory | `./output` |
| `--save-artifacts/--no-artifacts` | Save intermediate images | Disabled |
| `--signature-report/--no-signature-report` | Generate signature report | Enabled |
| `--parallel/--no-parallel` | Parallel page processing | Enabled |
| `--max-workers` | Maximum parallel workers | 3 |
| `--verbose`, `-v` | Verbose output | Disabled |

### Run ID System

Each run generates a unique **Run ID** in the format:
```
{document_name}_{provider}_{YYYYMMDD_HHMMSS}
```

Example: `supplementary_facility_letter_azure_20260204_164645`

All outputs are saved in a dedicated directory under `output/{run_id}/`.
