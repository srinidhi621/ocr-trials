"""
Base Provider Interface
Abstract base class defining the interface for OCR providers (Azure, Vertex).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class RegionType(Enum):
    """Types of regions that can be detected in a document."""
    TEXT_BLOCK = "text_block"
    TABLE = "table"
    SIGNATURE = "signature"
    REDACTED = "redacted"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


@dataclass
class BoundingBox:
    """Bounding box coordinates for a region."""
    x: float  # Top-left x
    y: float  # Top-left y
    width: float
    height: float
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class TableCell:
    """Represents a single cell in a table."""
    row_index: int
    column_index: int
    content: str
    row_span: int = 1
    column_span: int = 1
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None


@dataclass
class Table:
    """Represents a table extracted from a document."""
    cells: List[TableCell]
    row_count: int
    column_count: int
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    markdown: str = ""
    
    def to_grid(self) -> List[List[str]]:
        """Convert cells to a 2D grid representation."""
        grid = [["" for _ in range(self.column_count)] for _ in range(self.row_count)]
        for cell in self.cells:
            if cell.row_index < self.row_count and cell.column_index < self.column_count:
                for r in range(cell.row_index, min(self.row_count, cell.row_index + cell.row_span)):
                    for c in range(cell.column_index, min(self.column_count, cell.column_index + cell.column_span)):
                        if not grid[r][c]:
                            grid[r][c] = cell.content
        return grid
    
    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if self.markdown:
            return self.markdown
        
        grid = self.to_grid()
        if not grid:
            return ""
        
        lines = []
        # Header row
        lines.append("| " + " | ".join(grid[0]) + " |")
        # Separator
        lines.append("| " + " | ".join(["---"] * self.column_count) + " |")
        # Data rows
        for row in grid[1:]:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)


@dataclass
class SignatureBlock:
    """Represents a signature block in a document."""
    name: Optional[str] = None
    designation: Optional[str] = None
    date: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    page_number: int = 0
    confidence: float = 1.0
    visual_description: str = ""


@dataclass
class Region:
    """Represents a detected region in a document page."""
    region_type: RegionType
    bbox: BoundingBox
    content: str = ""
    confidence: float = 1.0
    table: Optional[Table] = None
    signature: Optional[SignatureBlock] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutResult:
    """Result of layout analysis for a single page."""
    page_number: int
    regions: List[Region]
    width: int
    height: int
    raw_text: str = ""
    confidence: float = 1.0
    
    def get_regions_by_type(self, region_type: RegionType) -> List[Region]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.region_type == region_type]


@dataclass
class PageExtraction:
    """Complete extraction result for a single page."""
    page_number: int
    layout: LayoutResult
    text_blocks: List[str] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    signatures: List[SignatureBlock] = field(default_factory=list)
    redactions: List[BoundingBox] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """Complete extraction result for a document."""
    source_file: str
    provider: str
    pages: List[PageExtraction]
    total_pages: int
    overall_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_tables(self) -> List[Table]:
        """Get all tables from all pages."""
        tables = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables
    
    def get_all_signatures(self) -> List[SignatureBlock]:
        """Get all signatures from all pages."""
        signatures = []
        for page in self.pages:
            signatures.extend(page.signatures)
        return signatures


class OCRProvider(ABC):
    """
    Abstract base class for OCR providers.
    
    Implementations should provide concrete methods for:
    - Layout analysis (detecting regions)
    - Text extraction
    - Table extraction
    - Multimodal LLM queries
    """
    
    def __init__(self, name: str):
        """Initialize the provider with a name identifier."""
        self.name = name
    
    @abstractmethod
    def analyze_layout(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze the layout of a document image.
        
        Args:
            image: Image bytes (PNG/JPEG)
            page_number: Page number for reference
            
        Returns:
            LayoutResult with detected regions
        """
        pass
    
    @abstractmethod
    def extract_text(self, image: bytes) -> str:
        """
        Extract all text from an image.
        
        Args:
            image: Image bytes (PNG/JPEG)
            
        Returns:
            Extracted text as string
        """
        pass
    
    @abstractmethod
    def extract_tables(self, image: bytes) -> List[Table]:
        """
        Extract tables from an image.
        
        Args:
            image: Image bytes (PNG/JPEG)
            
        Returns:
            List of Table objects
        """
        pass
    
    @abstractmethod
    def multimodal_query(self, image: bytes, prompt: str) -> str:
        """
        Send a multimodal query (image + text) to the LLM.
        
        Args:
            image: Image bytes (PNG/JPEG)
            prompt: Text prompt for the LLM
            
        Returns:
            LLM response as string
        """
        pass
    
    @abstractmethod
    def analyze_signature(self, image: bytes, context_image: Optional[bytes] = None) -> SignatureBlock:
        """
        Analyze a signature region.
        
        Args:
            image: Cropped signature image bytes
            context_image: Optional wider context including name/designation
            
        Returns:
            SignatureBlock with extracted information
        """
        pass
    
    def extract_page(self, image: bytes, page_number: int = 1) -> PageExtraction:
        """
        Extract all content from a page image.
        
        This is a convenience method that combines layout analysis
        with region-specific extraction.
        
        Args:
            image: Image bytes (PNG/JPEG)
            page_number: Page number for reference
            
        Returns:
            PageExtraction with all extracted content
        """
        # Get layout
        layout = self.analyze_layout(image, page_number)
        
        # Extract by region type
        text_blocks = []
        tables = []
        signatures = []
        redactions = []
        
        for region in layout.regions:
            if region.region_type == RegionType.TEXT_BLOCK:
                text_blocks.append(region.content)
            elif region.region_type == RegionType.TABLE:
                if region.table:
                    tables.append(region.table)
            elif region.region_type == RegionType.SIGNATURE:
                if region.signature:
                    signatures.append(region.signature)
            elif region.region_type == RegionType.REDACTED:
                redactions.append(region.bbox)
        
        return PageExtraction(
            page_number=page_number,
            layout=layout,
            text_blocks=text_blocks,
            tables=tables,
            signatures=signatures,
            redactions=redactions,
            confidence=layout.confidence
        )
