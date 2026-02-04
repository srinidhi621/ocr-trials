"""
Document OCR Pipeline Package
Multi-stage pipeline for extracting content from scanned PDFs.
"""

from .preprocessor import Preprocessor, PreprocessConfig
from .extractor import Extractor, ExtractorConfig
from .postprocessor import Postprocessor, SignatureComparator
from .reviewer import Reviewer, ConfidenceReport
from .signature_analyzer import (
    SignatureSnippet,
    SignatureGroup,
    SignaturePairComparison,
    SignatureAnalysisReport,
    SignatureAnalyzer,
)
from .signature_report import (
    SignatureReportGenerator,
    SignatureReportConfig,
    generate_signature_report,
)

__all__ = [
    # Core pipeline stages
    "Preprocessor",
    "PreprocessConfig",
    "Extractor",
    "ExtractorConfig",
    "Postprocessor",
    "SignatureComparator",
    "Reviewer",
    "ConfidenceReport",
    # Signature analysis
    "SignatureSnippet",
    "SignatureGroup",
    "SignaturePairComparison",
    "SignatureAnalysisReport",
    "SignatureAnalyzer",
    # Signature reports
    "SignatureReportGenerator",
    "SignatureReportConfig",
    "generate_signature_report",
]
