"""
OCR Providers Package
Provides abstraction layer for different OCR/Document Intelligence services.
"""

from .base import (
    OCRProvider,
    LayoutResult,
    RegionType,
    Region,
    Table,
    TableCell,
    SignatureBlock,
    ExtractionResult,
    PageExtraction,
)

__all__ = [
    "OCRProvider",
    "LayoutResult",
    "RegionType",
    "Region",
    "Table",
    "TableCell",
    "SignatureBlock",
    "ExtractionResult",
    "PageExtraction",
]
