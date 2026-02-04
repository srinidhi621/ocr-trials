"""
Stage 2: Extractor
Handles OCR and content extraction using providers.

Optimizations (Task 9):
- Batch signature validation: Multiple CV candidates in single LLM call
- Conditional LLM scan: Skip full-page LLM scan if signatures already found
- Tighter CV thresholds: Reduce false positives
- Batch table enhancement: Multiple tables in single LLM call per page
- Timing instrumentation: Per-stage timing metrics
"""

import os
import json
import time
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from .preprocessor import PageImage
from providers.base import (
    OCRProvider,
    LayoutResult,
    RegionType,
    Region,
    Table,
    SignatureBlock,
    BoundingBox,
    PageExtraction,
    ExtractionResult,
)

logger = logging.getLogger("ocr_pipeline")


# Prompt templates for LLM queries
TEXT_EXTRACTION_PROMPT = """Extract all text from this document region EXACTLY as written.

STRICT RULES:
1. Preserve all formatting (bold, italics if visible from context)
2. Preserve line breaks and paragraph structure
3. Preserve special characters and symbols exactly
4. Do NOT summarize or paraphrase - extract verbatim
5. Financial values must be character-for-character exact

Output the text in clean Markdown format."""

TABLE_EXTRACTION_PROMPT = """Convert this table to Markdown format.

STRICT RULES:
1. Preserve EXACT structure (rows/columns) as shown in the image
2. For merged cells, repeat the content or use appropriate representation
3. Copy ALL values VERBATIM - do not summarize or modify any values
4. Financial values must be character-for-character exact (e.g., "INR 500,000,000.00")
5. Do not add any extra rows or columns
6. If a cell is empty, leave it empty in the Markdown

Output ONLY the Markdown table, no explanation or additional text."""

SIGNATURE_ANALYSIS_PROMPT = """Analyze this signature block. Extract the following information:

1. **Name**: The printed name associated with the signature (look for text near/below the signature)
2. **Designation**: The job title or role (e.g., "Managing Director", "CFO", "Authorized Signatory")
3. **Date**: Any date visible near the signature
4. **Visual Description**: Brief description of the signature's visual characteristics for comparison purposes (e.g., "cursive with prominent loop at start", "initials JD with underline", "flowing script with flourish")

Return ONLY valid JSON with keys: name, designation, date, visual_description
If any field is not visible or cannot be determined, use null for that field.

Example output:
{"name": "John Smith", "designation": "Managing Director", "date": "15/01/2026", "visual_description": "cursive signature with prominent J loop"}"""


@dataclass
class ExtractorConfig:
    """Configuration for extraction operations."""
    extract_tables: bool = True
    extract_signatures: bool = True
    detect_redactions: bool = True
    use_llm_for_text: bool = True  # Use LLM for text extraction (more accurate)
    use_llm_for_tables: bool = True  # Use LLM to enhance table Markdown
    signature_context_padding: int = 50  # Pixels to add around signature for context
    save_signatures: bool = True  # Whether to save signature images
    output_dir: Optional[str] = None
    
    # Signature detection settings
    signature_detection_method: str = "hybrid"  # "llm", "cv", "hybrid"
    signature_min_width: int = 30   # Minimum signature width in pixels
    signature_max_width: int = 1200  # Maximum signature width
    signature_min_height: int = 10  # Minimum signature height
    signature_max_height: int = 300 # Maximum signature height
    signature_aspect_ratio_min: float = 1.5  # Tightened: was 1.0 (Task 9.3)
    signature_aspect_ratio_max: float = 20.0
    signature_page_region: str = "lower_half"  # Default region if search list not provided
    signature_search_regions: List[str] = field(default_factory=lambda: ["lower_half", "full"])
    signature_ink_density_min: float = 0.03  # Tightened: was 0.01 (Task 9.3)
    signature_ink_density_max: float = 0.85  # Maximum ink density (avoid solid blocks)
    signature_fill_ratio_min: float = 0.15  # Tightened: was 0.1 (Task 9.3)
    signature_fill_ratio_max: float = 0.98
    signature_variance_min: float = 50.0
    signature_min_width_px: int = 100  # New: Skip regions smaller than typical signature (Task 9.3)
    
    # Performance optimization settings (Task 9)
    skip_llm_scan_if_found: bool = True  # Task 9.2: Skip full-page LLM scan if DI/CV found signatures
    batch_signature_validation: bool = True  # Task 9.1: Batch CV candidates into single LLM call
    batch_signature_max_candidates: int = 8  # Max candidates per batch to avoid token limits
    batch_table_enhancement: bool = True  # Task 9.5: Batch multiple tables in single LLM call
    batch_table_max_tables: int = 4  # Max tables per batch
    skip_text_overlap_regions: bool = True  # Task 9.3: Skip CV regions overlapping with text blocks


@dataclass
class TimingMetrics:
    """Timing metrics for pipeline stages."""
    di_layout_ms: float = 0.0
    cv_signature_detection_ms: float = 0.0
    signature_validation_ms: float = 0.0
    signature_validation_count: int = 0
    llm_signature_scan_ms: float = 0.0
    llm_signature_scan_skipped: bool = False
    table_enhancement_ms: float = 0.0
    table_enhancement_count: int = 0
    redaction_detection_ms: float = 0.0
    total_page_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "di_layout_ms": round(self.di_layout_ms, 1),
            "cv_signature_detection_ms": round(self.cv_signature_detection_ms, 1),
            "signature_validation_ms": round(self.signature_validation_ms, 1),
            "signature_validation_count": self.signature_validation_count,
            "llm_signature_scan_ms": round(self.llm_signature_scan_ms, 1),
            "llm_signature_scan_skipped": self.llm_signature_scan_skipped,
            "table_enhancement_ms": round(self.table_enhancement_ms, 1),
            "table_enhancement_count": self.table_enhancement_count,
            "redaction_detection_ms": round(self.redaction_detection_ms, 1),
            "total_page_ms": round(self.total_page_ms, 1),
        }


class Extractor:
    """
    Stage 2: Extractor
    
    Handles:
    1. Layout analysis using document intelligence services
    2. Region-specific OCR (text blocks, tables, signatures)
    3. Redaction detection
    4. LLM-enhanced extraction for accuracy
    
    Optimizations (Task 9):
    - Batch signature validation reduces LLM calls
    - Conditional LLM scan skips unnecessary full-page scans
    - Tighter CV thresholds reduce false positives
    - Batch table enhancement reduces LLM calls
    - Timing instrumentation tracks performance
    """
    
    def __init__(self, provider: OCRProvider, config: Optional[ExtractorConfig] = None):
        """
        Initialize extractor with a provider.
        
        Args:
            provider: OCR provider (Azure or Vertex)
            config: Extraction configuration
        """
        import threading
        
        self.provider = provider
        self.config = config or ExtractorConfig()
        self._signature_counter = 0
        self._signature_lock = threading.Lock()  # Thread-safe counter for parallel processing
        self._timing_metrics: Dict[int, TimingMetrics] = {}  # Per-page timing
    
    def extract_document(
        self,
        pages: List[PageImage],
        source_file: str,
        parallel: bool = False,
        max_workers: int = 3
    ) -> ExtractionResult:
        """
        Extract content from all pages of a document.
        
        Task 9.4: Supports parallel page processing for reduced latency.
        
        Args:
            pages: List of preprocessed page images
            source_file: Original source file path
            parallel: Whether to process pages in parallel (Task 9.4)
            max_workers: Maximum parallel workers (default 3 for rate limiting)
            
        Returns:
            ExtractionResult with all extracted content
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        doc_start = time.time()
        page_extractions = []
        
        if parallel and len(pages) > 1:
            # Task 9.4: Parallel page processing
            logger.info(f"Processing {len(pages)} pages in parallel (max_workers={max_workers})")
            
            # Use ThreadPoolExecutor for I/O-bound operations (API calls)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all pages for processing
                future_to_page = {
                    executor.submit(self.extract_page, page): page.page_number
                    for page in pages
                }
                
                # Collect results as they complete
                results_by_page = {}
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        extraction = future.result()
                        results_by_page[page_num] = extraction
                    except Exception as e:
                        logger.error(f"Page {page_num} extraction failed: {e}")
                        # Create empty extraction for failed page
                        results_by_page[page_num] = PageExtraction(
                            page_number=page_num,
                            layout=LayoutResult(page_num, [], 0, 0),
                            confidence=0.0
                        )
                
                # Sort by page number to maintain order
                for page_num in sorted(results_by_page.keys()):
                    page_extractions.append(results_by_page[page_num])
        else:
            # Sequential processing (original behavior)
            for page in pages:
                extraction = self.extract_page(page)
                page_extractions.append(extraction)
        
        doc_elapsed = time.time() - doc_start
        
        # Calculate overall confidence
        confidences = [p.confidence for p in page_extractions]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Log summary timing
        total_timing = self._aggregate_timing_metrics()
        logger.info(
            f"Document extraction complete: {len(pages)} pages in {doc_elapsed:.1f}s "
            f"(avg {doc_elapsed/len(pages)*1000:.0f}ms/page)"
        )
        if total_timing:
            logger.info(
                f"  Timing breakdown: DI={total_timing.get('di_layout_ms', 0):.0f}ms, "
                f"CV+Sig={total_timing.get('signature_total_ms', 0):.0f}ms, "
                f"Tables={total_timing.get('table_enhancement_ms', 0):.0f}ms"
            )
        
        result = ExtractionResult(
            source_file=source_file,
            provider=self.provider.name,
            pages=page_extractions,
            total_pages=len(pages),
            overall_confidence=overall_confidence
        )
        
        # Add timing metadata
        result.metadata['timing'] = self.get_timing_metrics()
        result.metadata['timing_summary'] = total_timing
        result.metadata['parallel_processing'] = parallel
        result.metadata['extraction_time_seconds'] = doc_elapsed
        
        return result
    
    def _aggregate_timing_metrics(self) -> Dict[str, float]:
        """Aggregate timing metrics across all pages."""
        if not self._timing_metrics:
            return {}
        
        total = {
            'di_layout_ms': 0.0,
            'cv_signature_detection_ms': 0.0,
            'signature_validation_ms': 0.0,
            'signature_validation_count': 0,
            'llm_signature_scan_ms': 0.0,
            'llm_scans_skipped': 0,
            'table_enhancement_ms': 0.0,
            'table_enhancement_count': 0,
            'redaction_detection_ms': 0.0,
            'total_page_ms': 0.0,
        }
        
        for timing in self._timing_metrics.values():
            total['di_layout_ms'] += timing.di_layout_ms
            total['cv_signature_detection_ms'] += timing.cv_signature_detection_ms
            total['signature_validation_ms'] += timing.signature_validation_ms
            total['signature_validation_count'] += timing.signature_validation_count
            total['llm_signature_scan_ms'] += timing.llm_signature_scan_ms
            if timing.llm_signature_scan_skipped:
                total['llm_scans_skipped'] += 1
            total['table_enhancement_ms'] += timing.table_enhancement_ms
            total['table_enhancement_count'] += timing.table_enhancement_count
            total['redaction_detection_ms'] += timing.redaction_detection_ms
            total['total_page_ms'] += timing.total_page_ms
        
        # Add computed totals
        total['signature_total_ms'] = (
            total['cv_signature_detection_ms'] + 
            total['signature_validation_ms'] + 
            total['llm_signature_scan_ms']
        )
        
        return total
    
    def extract_page(self, page: PageImage) -> PageExtraction:
        """
        Extract content from a single page.
        
        Args:
            page: Preprocessed page image
            
        Returns:
            PageExtraction with all extracted content
        """
        page_start = time.time()
        timing = TimingMetrics()
        
        # Get image bytes for provider
        image_bytes = page.to_bytes()
        
        # Step 1: Analyze layout to get regions (Task 9.6 timing)
        di_start = time.time()
        layout = self.provider.analyze_layout(image_bytes, page.page_number)
        timing.di_layout_ms = (time.time() - di_start) * 1000
        logger.debug(f"Page {page.page_number} - DI Layout: {timing.di_layout_ms:.1f}ms")
        
        # Step 2: Detect redactions (if configured)
        redactions = []
        if self.config.detect_redactions:
            redact_start = time.time()
            redactions = self._detect_redactions(page.image)
            timing.redaction_detection_ms = (time.time() - redact_start) * 1000
        
        # Step 3: Detect signatures (may not be in layout)
        signatures = []
        if self.config.extract_signatures:
            signatures = self._extract_signatures_optimized(page, layout, image_bytes, timing)
        
        # Step 4: Process tables (with batching if configured)
        tables = []
        if self.config.extract_tables:
            tables = self._extract_tables_optimized(page, layout, image_bytes, timing)
        
        # Step 5: Extract text blocks
        text_blocks = self._extract_text_blocks(layout, image_bytes)
        
        # Calculate page confidence
        confidence = layout.confidence
        
        # Store timing metrics
        timing.total_page_ms = (time.time() - page_start) * 1000
        self._timing_metrics[page.page_number] = timing
        
        logger.info(
            f"Page {page.page_number} complete: "
            f"DI={timing.di_layout_ms:.0f}ms, "
            f"CV+Sig={timing.cv_signature_detection_ms + timing.signature_validation_ms:.0f}ms "
            f"({timing.signature_validation_count} candidates), "
            f"LLM_scan={'skipped' if timing.llm_signature_scan_skipped else f'{timing.llm_signature_scan_ms:.0f}ms'}, "
            f"Tables={timing.table_enhancement_ms:.0f}ms ({timing.table_enhancement_count}), "
            f"Total={timing.total_page_ms:.0f}ms"
        )
        
        return PageExtraction(
            page_number=page.page_number,
            layout=layout,
            text_blocks=text_blocks,
            tables=tables,
            signatures=signatures,
            redactions=redactions,
            confidence=confidence
        )
    
    def get_timing_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get timing metrics for all processed pages."""
        return {page: timing.to_dict() for page, timing in self._timing_metrics.items()}
    
    def _extract_text_blocks(
        self,
        layout: LayoutResult,
        image_bytes: bytes
    ) -> List[str]:
        """
        Extract text blocks from layout.
        
        Args:
            layout: Layout analysis result
            image_bytes: Full page image bytes
            
        Returns:
            List of extracted text strings
        """
        text_blocks = []
        
        for region in layout.regions:
            if region.region_type == RegionType.TEXT_BLOCK:
                if self.config.use_llm_for_text and region.content:
                    # Use document intelligence result
                    text_blocks.append(region.content)
                elif region.content:
                    text_blocks.append(region.content)
        
        # If no text blocks from regions, use raw text
        if not text_blocks and layout.raw_text:
            text_blocks.append(layout.raw_text)
        
        return text_blocks
    
    def _extract_tables(
        self,
        page: PageImage,
        layout: LayoutResult,
        image_bytes: bytes
    ) -> List[Table]:
        """
        Extract tables with enhanced Markdown using LLM.
        
        Args:
            page: Page image
            layout: Layout analysis result
            image_bytes: Full page image bytes
            
        Returns:
            List of Table objects with Markdown
        """
        tables = []
        
        for region in layout.regions:
            if region.region_type == RegionType.TABLE and region.table:
                table = region.table
                
                # Enhance with LLM if configured
                if self.config.use_llm_for_tables:
                    # Crop table region for more focused analysis
                    bbox = self._bbox_to_pixels(region.bbox, layout, page)
                    table_image = self._crop_region(page.image, bbox)
                    if table_image is not None:
                        table_bytes = self._image_to_bytes(table_image)
                        
                        # Use LLM to generate better Markdown
                        try:
                            markdown = self.provider.multimodal_query(
                                table_bytes,
                                TABLE_EXTRACTION_PROMPT
                            )
                            # Extract just the table from response
                            extracted = self._extract_markdown_table(markdown)
                            if self._markdown_table_matches_dimensions(
                                extracted,
                                table.row_count,
                                table.column_count
                            ):
                                table.markdown = extracted
                            else:
                                table.markdown = table.to_markdown()
                        except Exception as e:
                            # Fall back to basic conversion
                            table.markdown = table.to_markdown()
                    else:
                        table.markdown = table.to_markdown()
                else:
                    table.markdown = table.to_markdown()
                
                tables.append(table)
        
        return tables
    
    def _extract_tables_optimized(
        self,
        page: PageImage,
        layout: LayoutResult,
        image_bytes: bytes,
        timing: TimingMetrics
    ) -> List[Table]:
        """
        Extract tables with optimized batch LLM enhancement.
        
        Task 9.5: Batch multiple tables in a single LLM call to reduce latency.
        
        Args:
            page: Page image
            layout: Layout analysis result
            image_bytes: Full page image bytes
            timing: TimingMetrics to update
            
        Returns:
            List of Table objects with Markdown
        """
        # Collect all tables from layout
        table_regions = []
        for region in layout.regions:
            if region.region_type == RegionType.TABLE and region.table:
                table_regions.append((region, region.table))
        
        timing.table_enhancement_count = len(table_regions)
        
        if not table_regions:
            return []
        
        tables = []
        
        # If LLM enhancement is disabled, just convert to markdown
        if not self.config.use_llm_for_tables:
            for region, table in table_regions:
                table.markdown = table.to_markdown()
                tables.append(table)
            return tables
        
        enhance_start = time.time()
        
        # If batch enhancement is disabled or only one table, use individual enhancement
        if not self.config.batch_table_enhancement or len(table_regions) == 1:
            for region, table in table_regions:
                bbox = self._bbox_to_pixels(region.bbox, layout, page)
                table_image = self._crop_region(page.image, bbox)
                
                if table_image is not None:
                    table_bytes = self._image_to_bytes(table_image)
                    try:
                        markdown = self.provider.multimodal_query(
                            table_bytes,
                            TABLE_EXTRACTION_PROMPT
                        )
                        extracted = self._extract_markdown_table(markdown)
                        if self._markdown_table_matches_dimensions(
                            extracted,
                            table.row_count,
                            table.column_count
                        ):
                            table.markdown = extracted
                        else:
                            table.markdown = table.to_markdown()
                    except Exception:
                        table.markdown = table.to_markdown()
                else:
                    table.markdown = table.to_markdown()
                
                tables.append(table)
            
            timing.table_enhancement_ms = (time.time() - enhance_start) * 1000
            return tables
        
        # Batch enhancement: process multiple tables in single LLM call
        max_batch = self.config.batch_table_max_tables
        
        for batch_start in range(0, len(table_regions), max_batch):
            batch = table_regions[batch_start:batch_start + max_batch]
            
            # Create montage of table images
            table_crops = []
            table_infos = []
            
            for region, table in batch:
                bbox = self._bbox_to_pixels(region.bbox, layout, page)
                crop = self._crop_region(page.image, bbox)
                
                if crop is not None and crop.size > 0:
                    table_crops.append(crop)
                    table_infos.append({
                        'table': table,
                        'rows': table.row_count,
                        'cols': table.column_count
                    })
                else:
                    # No crop available, use DI markdown
                    table.markdown = table.to_markdown()
                    tables.append(table)
            
            if not table_crops:
                continue
            
            # Create vertical stack of table images
            montage = self._create_table_montage(table_crops)
            if montage is None:
                # Fallback to individual enhancement
                for info in table_infos:
                    info['table'].markdown = info['table'].to_markdown()
                    tables.append(info['table'])
                continue
            
            montage_bytes = self._image_to_bytes(montage)
            
            # Batch table enhancement prompt
            batch_prompt = f"""This image contains {len(table_crops)} tables stacked vertically, separated by red lines.

For EACH table (numbered 1 to {len(table_crops)} from top to bottom), convert it to Markdown format.

STRICT RULES:
1. Preserve EXACT structure (rows/columns) as shown in each table image
2. Copy ALL values VERBATIM - do not summarize or modify any values
3. Financial values must be character-for-character exact (e.g., "INR 500,000,000.00")
4. For merged cells, repeat the content appropriately
5. If a cell is empty, leave it empty in the Markdown

Expected dimensions for each table:
{chr(10).join(f"Table {i+1}: {info['rows']} rows × {info['cols']} columns" for i, info in enumerate(table_infos))}

Return your response in this exact format:
---TABLE 1---
| header1 | header2 |
|---|---|
| data | data |

---TABLE 2---
...

Output ONLY the markdown tables with the separators, no explanations."""

            try:
                response = self.provider.multimodal_query(montage_bytes, batch_prompt)
                
                # Parse batch response
                table_markdowns = self._parse_batch_table_response(response, len(table_infos))
                
                for i, info in enumerate(table_infos):
                    if i < len(table_markdowns) and table_markdowns[i]:
                        extracted = self._extract_markdown_table(table_markdowns[i])
                        # Validate dimensions (accuracy check - never sacrifice for performance)
                        if self._markdown_table_matches_dimensions(
                            extracted,
                            info['rows'],
                            info['cols']
                        ):
                            info['table'].markdown = extracted
                        else:
                            # Dimension mismatch - use DI result for accuracy
                            logger.debug(f"Table {i+1} dimension mismatch in batch, using DI markdown")
                            info['table'].markdown = info['table'].to_markdown()
                    else:
                        info['table'].markdown = info['table'].to_markdown()
                    
                    tables.append(info['table'])
                    
            except Exception as e:
                logger.warning(f"Batch table enhancement failed: {e}, falling back to individual")
                # Fallback to individual enhancement
                for info in table_infos:
                    info['table'].markdown = info['table'].to_markdown()
                    tables.append(info['table'])
        
        timing.table_enhancement_ms = (time.time() - enhance_start) * 1000
        return tables
    
    def _create_table_montage(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create a vertical stack of table images with separators.
        
        Args:
            images: List of table crop images
            
        Returns:
            Stacked montage image or None
        """
        import cv2
        
        if not images:
            return None
        
        # Find max width
        max_width = max(img.shape[1] for img in images)
        separator_height = 10
        
        # Calculate total height
        total_height = sum(img.shape[0] for img in images) + separator_height * (len(images) - 1)
        
        # Create canvas (white background)
        montage = np.full((total_height, max_width, 3), 255, dtype=np.uint8)
        
        y_offset = 0
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            
            # Center image horizontally
            x_offset = (max_width - w) // 2
            montage[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
            y_offset += h
            
            # Add red separator line (except after last image)
            if i < len(images) - 1:
                cv2.line(montage, (0, y_offset + separator_height // 2), 
                        (max_width, y_offset + separator_height // 2), (0, 0, 255), 3)
                y_offset += separator_height
        
        return montage
    
    def _parse_batch_table_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse batch table enhancement response.
        
        Args:
            response: LLM response with multiple tables
            expected_count: Expected number of tables
            
        Returns:
            List of markdown strings for each table
        """
        import re
        
        # Split by table separator pattern
        parts = re.split(r'---TABLE\s*\d+---', response, flags=re.IGNORECASE)
        
        # First part is usually empty or preamble, skip it
        table_markdowns = []
        for part in parts[1:]:  # Skip first (empty/preamble)
            part = part.strip()
            if part:
                table_markdowns.append(part)
        
        # If splitting didn't work well, try to find tables by | patterns
        if len(table_markdowns) < expected_count:
            # Find all markdown table blocks
            lines = response.split('\n')
            current_table = []
            tables_found = []
            
            for line in lines:
                if line.strip().startswith('|'):
                    current_table.append(line)
                elif current_table:
                    # End of current table
                    if len(current_table) >= 2:  # At least header + separator
                        tables_found.append('\n'.join(current_table))
                    current_table = []
            
            # Don't forget last table
            if current_table and len(current_table) >= 2:
                tables_found.append('\n'.join(current_table))
            
            if len(tables_found) >= expected_count:
                table_markdowns = tables_found[:expected_count]
        
        return table_markdowns
    
    def _extract_signatures_optimized(
        self,
        page: PageImage,
        layout: LayoutResult,
        image_bytes: bytes,
        timing: TimingMetrics
    ) -> List[SignatureBlock]:
        """
        Extract and analyze signatures using optimized hybrid detection.
        
        Optimizations (Task 9):
        - Batch CV candidates into single LLM call (9.1)
        - Skip full-page LLM scan if signatures already found (9.2)
        - Tighter CV thresholds with text overlap filter (9.3)
        
        Args:
            page: Page image
            layout: Layout analysis result
            image_bytes: Full page image bytes
            timing: TimingMetrics to update
            
        Returns:
            List of SignatureBlock objects
        """
        signatures = []
        detected_regions: List[BoundingBox] = []
        
        # Method selection based on config
        method = self.config.signature_detection_method
        
        # Get text block regions for overlap filtering (Task 9.3)
        text_regions = []
        if self.config.skip_text_overlap_regions:
            for region in layout.regions:
                if region.region_type == RegionType.TEXT_BLOCK and region.bbox:
                    text_regions.append(self._bbox_to_pixels(region.bbox, layout, page) or region.bbox)
        
        # Step 1: Check layout for signature regions (from Document Intelligence)
        for region in layout.regions:
            if region.region_type == RegionType.SIGNATURE:
                sig = self._analyze_signature_region(page, region, layout)
                if sig:
                    signatures.append(sig)
                    if region.bbox:
                        detected_regions.append(self._bbox_to_pixels(region.bbox, layout, page) or region.bbox)
        
        di_found_signatures = len(signatures) > 0
        
        # Step 2: CV-based detection (if method is "cv" or "hybrid")
        cv_candidates = []
        if method in ("cv", "hybrid"):
            cv_start = time.time()
            cv_regions = self._detect_signatures_cv(page.image)
            timing.cv_signature_detection_ms = (time.time() - cv_start) * 1000
            
            # Filter CV regions (Task 9.3)
            for bbox in cv_regions:
                # Skip if overlaps with already detected signatures
                if self._overlaps_with_any(bbox, detected_regions, threshold=0.3):
                    continue
                
                # Skip if overlaps significantly with text blocks (Task 9.3)
                if self.config.skip_text_overlap_regions:
                    if self._overlaps_with_any(bbox, text_regions, threshold=0.5):
                        continue
                
                # Skip regions smaller than minimum signature width (Task 9.3)
                if bbox.width < self.config.signature_min_width_px:
                    continue
                
                cv_candidates.append(bbox)
            
            timing.signature_validation_count = len(cv_candidates)
            
            # Validate CV candidates (with batching if enabled - Task 9.1)
            if cv_candidates:
                val_start = time.time()
                if self.config.batch_signature_validation and len(cv_candidates) > 1:
                    # Batch validation: process multiple candidates in single LLM call
                    validated_sigs = self._batch_validate_signatures(page, cv_candidates, image_bytes)
                else:
                    # Individual validation (original behavior)
                    validated_sigs = []
                    for bbox in cv_candidates:
                        sig = self._analyze_cv_signature_region(page, bbox, image_bytes)
                        if sig:
                            validated_sigs.append(sig)
                
                for sig in validated_sigs:
                    signatures.append(sig)
                    if sig.bbox:
                        detected_regions.append(sig.bbox)
                
                timing.signature_validation_ms = (time.time() - val_start) * 1000
        
        cv_found_signatures = len(signatures) > (1 if di_found_signatures else 0)
        
        # Step 3: LLM-based detection (if method is "llm" or "hybrid")
        # Task 9.2: Skip full-page LLM scan if signatures already found by DI or CV
        if method in ("llm", "hybrid"):
            should_skip_llm_scan = (
                self.config.skip_llm_scan_if_found 
                and (di_found_signatures or cv_found_signatures)
            )
            
            if should_skip_llm_scan:
                timing.llm_signature_scan_skipped = True
                logger.debug(f"Page {page.page_number} - Skipping LLM signature scan (already found {len(signatures)} signatures)")
            else:
                llm_start = time.time()
                llm_signatures = self._detect_signatures_with_llm(page, image_bytes)
                timing.llm_signature_scan_ms = (time.time() - llm_start) * 1000
                
                for sig in llm_signatures:
                    if sig.bbox and self._overlaps_with_any(sig.bbox, detected_regions, threshold=0.3):
                        continue
                    signatures.append(sig)
                    if sig.bbox:
                        detected_regions.append(sig.bbox)
        
        return signatures
    
    def _batch_validate_signatures(
        self,
        page: PageImage,
        candidates: List[BoundingBox],
        image_bytes: bytes
    ) -> List[SignatureBlock]:
        """
        Batch validate multiple signature candidates in a single LLM call.
        
        Task 9.1: Instead of N individual LLM calls, create a montage image
        with all candidates and ask LLM to identify which are actual signatures.
        
        Args:
            page: Page image
            candidates: List of CV-detected candidate regions
            image_bytes: Full page image bytes
            
        Returns:
            List of validated SignatureBlock objects
        """
        import cv2
        
        if not candidates:
            return []
        
        validated_signatures = []
        max_batch = self.config.batch_signature_max_candidates
        
        # Process in batches to avoid token limits
        for batch_start in range(0, len(candidates), max_batch):
            batch = candidates[batch_start:batch_start + max_batch]
            
            # Create montage of candidate regions
            cropped_images = []
            valid_indices = []
            
            for i, bbox in enumerate(batch):
                crop = self._crop_region(
                    page.image,
                    bbox,
                    padding=self.config.signature_context_padding
                )
                if crop is not None and crop.size > 0:
                    cropped_images.append(crop)
                    valid_indices.append(i)
            
            if not cropped_images:
                continue
            
            # Create grid montage
            montage, grid_info = self._create_signature_montage(cropped_images)
            if montage is None:
                # Fallback to individual validation
                for i, idx in enumerate(valid_indices):
                    sig = self._analyze_cv_signature_region(page, batch[idx], image_bytes)
                    if sig:
                        validated_signatures.append(sig)
                continue
            
            montage_bytes = self._image_to_bytes(montage)
            
            # Batch validation prompt
            validation_prompt = f"""Analyze this grid of {len(cropped_images)} image regions (numbered 1 to {len(cropped_images)}, left-to-right, top-to-bottom).

For EACH numbered region, determine if it contains a handwritten signature.

Return JSON array with one object per region:
[
  {{"region": 1, "is_signature": true/false, "name": "...", "designation": "...", "date": "...", "visual_description": "..."}},
  ...
]

Rules:
- is_signature: true ONLY for actual handwritten signatures (not stamps, logos, printed text)
- If is_signature is false, other fields can be null
- name/designation/date: extract if visible near the signature region
- visual_description: brief description of signature style

Be conservative - if uncertain, set is_signature to false."""

            try:
                response = self.provider.multimodal_query(montage_bytes, validation_prompt)
                
                # Parse JSON response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    results = json.loads(response[json_start:json_end])
                else:
                    # Fallback to individual validation
                    for i, idx in enumerate(valid_indices):
                        sig = self._analyze_cv_signature_region(page, batch[idx], image_bytes)
                        if sig:
                            validated_signatures.append(sig)
                    continue
                
                # Process results
                for result in results:
                    region_num = result.get('region', 0) - 1  # Convert to 0-indexed
                    if region_num < 0 or region_num >= len(valid_indices):
                        continue
                    
                    if not result.get('is_signature', False):
                        continue
                    
                    orig_idx = valid_indices[region_num]
                    bbox = batch[orig_idx]
                    
                    sig_image = self._crop_region(page.image, bbox, padding=10)
                    
                    sig_block = SignatureBlock(
                        name=result.get('name'),
                        designation=result.get('designation'),
                        date=result.get('date'),
                        visual_description=result.get('visual_description', ''),
                        page_number=page.page_number,
                        bbox=bbox,
                        image=sig_image,
                        confidence=0.85  # Batch-validated
                    )
                    
                    # Save signature image if configured
                    if self.config.save_signatures and self.config.output_dir and sig_image is not None:
                        with self._signature_lock:
                            sig_block.image_path = self._save_signature_image(
                                sig_image,
                                page.page_number,
                                self._signature_counter
                            )
                            self._signature_counter += 1
                    
                    validated_signatures.append(sig_block)
                    
            except Exception as e:
                logger.warning(f"Batch signature validation failed: {e}, falling back to individual validation")
                # Fallback to individual validation on error
                for i, idx in enumerate(valid_indices):
                    sig = self._analyze_cv_signature_region(page, batch[idx], image_bytes)
                    if sig:
                        validated_signatures.append(sig)
        
        return validated_signatures
    
    def _create_signature_montage(
        self,
        images: List[np.ndarray],
        max_width: int = 1200
    ) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Create a grid montage from multiple signature candidate images.
        
        Args:
            images: List of cropped candidate images
            max_width: Maximum width for the montage
            
        Returns:
            Tuple of (montage image, list of (row, col, x, y) for each image)
        """
        import cv2
        
        if not images:
            return None, []
        
        # Normalize image sizes
        target_height = 80  # Reasonable height for signature crops
        resized = []
        for img in images:
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized.append(cv2.resize(img, (new_w, target_height)))
        
        # Calculate grid layout
        num_images = len(resized)
        cols = min(4, num_images)  # Max 4 columns
        rows = math.ceil(num_images / cols)
        
        # Find max width per column
        col_widths = [0] * cols
        for i, img in enumerate(resized):
            col = i % cols
            col_widths[col] = max(col_widths[col], img.shape[1])
        
        total_width = sum(col_widths) + (cols - 1) * 10  # 10px padding between columns
        total_height = rows * target_height + (rows - 1) * 10
        
        # Create montage canvas (white background)
        montage = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
        
        # Place images and add numbering
        grid_info = []
        for i, img in enumerate(resized):
            row = i // cols
            col = i % cols
            
            # Calculate position
            x = sum(col_widths[:col]) + col * 10
            y = row * (target_height + 10)
            
            # Place image
            h, w = img.shape[:2]
            montage[y:y+h, x:x+w] = img
            
            # Add number label
            cv2.putText(
                montage,
                str(i + 1),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            
            grid_info.append((row, col, x, y))
        
        return montage, grid_info
    
    def _extract_signatures(
        self,
        page: PageImage,
        layout: LayoutResult,
        image_bytes: bytes
    ) -> List[SignatureBlock]:
        """
        Extract and analyze signatures using hybrid detection.
        
        Uses a combination of:
        1. Layout analysis (from Document Intelligence)
        2. Computer vision-based detection (contour analysis)
        3. LLM-based detection (multimodal analysis)
        
        Args:
            page: Page image
            layout: Layout analysis result
            image_bytes: Full page image bytes
            
        Returns:
            List of SignatureBlock objects
        """
        signatures = []
        detected_regions: List[BoundingBox] = []
        
        # Method selection based on config
        method = self.config.signature_detection_method
        
        # Step 1: Check layout for signature regions (from Document Intelligence)
        for region in layout.regions:
            if region.region_type == RegionType.SIGNATURE:
                sig = self._analyze_signature_region(page, region, layout)
                if sig:
                    signatures.append(sig)
                    if region.bbox:
                        detected_regions.append(self._bbox_to_pixels(region.bbox, layout, page) or region.bbox)
        
        # Step 2: CV-based detection (if method is "cv" or "hybrid")
        if method in ("cv", "hybrid"):
            cv_regions = self._detect_signatures_cv(page.image)
            
            for bbox in cv_regions:
                # Skip if this region overlaps significantly with already detected signatures
                if self._overlaps_with_any(bbox, detected_regions, threshold=0.3):
                    continue
                
                # Analyze the detected region
                sig = self._analyze_cv_signature_region(page, bbox, image_bytes)
                if sig:
                    signatures.append(sig)
                    detected_regions.append(bbox)
        
        # Step 3: LLM-based detection (if method is "llm" or "hybrid")
        if method in ("llm", "hybrid"):
            llm_signatures = self._detect_signatures_with_llm(page, image_bytes)
            for sig in llm_signatures:
                if sig.bbox and self._overlaps_with_any(sig.bbox, detected_regions, threshold=0.3):
                    continue
                signatures.append(sig)
                if sig.bbox:
                    detected_regions.append(sig.bbox)
        
        return signatures
    
    def _detect_signatures_cv(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect potential signature regions using computer vision.
        
        Uses contour analysis with heuristics:
        - Signatures are typically in the lower portion of the page
        - Aspect ratio is wider than tall (1.5:1 to 12:1)
        - Moderate ink density (not solid blocks like redactions)
        - Specific size range
        
        Args:
            image: Page image as numpy array (BGR)
            
        Returns:
            List of BoundingBox objects for potential signature regions
        """
        import cv2
        
        height, width = image.shape[:2]
        
        # Determine search regions based on config
        search_regions = self.config.signature_search_regions or [self.config.signature_page_region]
        search_y_starts = []
        for region in search_regions:
            if region == "lower_third":
                search_y_starts.append(int(height * 2 / 3))
            elif region == "lower_half":
                search_y_starts.append(int(height / 2))
            else:  # "full"
                search_y_starts.append(0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        # Use OTSU for automatic threshold selection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Also try adaptive threshold for handwritten signatures
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # Combine both thresholding methods
        combined = cv2.bitwise_or(binary, adaptive)
        
        # Morphological operations to connect signature strokes
        # Use horizontal kernel to connect signature strokes
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        
        # Dilate horizontally to connect cursive strokes
        dilated = cv2.dilate(combined, kernel_h, iterations=2)
        # Slight vertical dilation
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signature_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip regions outside any search area
            if all(y < start for start in search_y_starts):
                continue
            
            # Apply size filters
            if w < self.config.signature_min_width or w > self.config.signature_max_width:
                continue
            if h < self.config.signature_min_height or h > self.config.signature_max_height:
                continue
            
            # Check aspect ratio (signatures are wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.config.signature_aspect_ratio_min:
                continue
            if aspect_ratio > self.config.signature_aspect_ratio_max:
                continue
            
            # Calculate ink density in the original binary image
            roi_binary = binary[y:y+h, x:x+w]
            if roi_binary.size == 0:
                continue
            
            ink_pixels = np.count_nonzero(roi_binary)
            total_pixels = roi_binary.size
            ink_density = ink_pixels / total_pixels
            
            # Filter by ink density
            if ink_density < self.config.signature_ink_density_min:
                continue
            if ink_density > self.config.signature_ink_density_max:
                continue
            
            # Additional check: signatures should not be perfectly rectangular
            contour_area = cv2.contourArea(contour)
            rect_area = w * h
            fill_ratio = contour_area / rect_area if rect_area > 0 else 0
            
            if fill_ratio < self.config.signature_fill_ratio_min or fill_ratio > self.config.signature_fill_ratio_max:
                continue
            
            # Check if there's variance in the region (not a solid block)
            roi_gray = gray[y:y+h, x:x+w]
            variance = np.var(roi_gray)
            if variance < self.config.signature_variance_min:
                continue
            
            signature_regions.append(BoundingBox(
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h)
            ))
        
        # Sort by y-coordinate (top to bottom) and de-duplicate overlaps
        signature_regions.sort(key=lambda b: b.y)
        signature_regions = self._dedupe_regions(signature_regions, threshold=0.3)
        
        return signature_regions
    
    def _analyze_cv_signature_region(
        self,
        page: PageImage,
        bbox: BoundingBox,
        image_bytes: bytes
    ) -> Optional[SignatureBlock]:
        """
        Analyze a CV-detected signature region using LLM.
        
        Args:
            page: Page image
            bbox: Detected signature bounding box
            image_bytes: Full page image bytes
            
        Returns:
            SignatureBlock or None if not a valid signature
        """
        # Crop the signature region with context padding
        sig_image = self._crop_region(
            page.image,
            bbox,
            padding=self.config.signature_context_padding
        )
        
        if sig_image is None:
            return None
        
        sig_bytes = self._image_to_bytes(sig_image)
        
        # Use LLM to analyze and validate the signature
        validation_prompt = """Analyze this image region. Is this a handwritten signature?

If YES, extract:
1. Name: The printed name near the signature (if visible)
2. Designation: Job title or role (if visible)
3. Date: Any date near the signature (if visible)
4. Visual Description: Brief description of the signature style

Return JSON: {"is_signature": true/false, "name": "...", "designation": "...", "date": "...", "visual_description": "..."}

If this is NOT a signature (e.g., printed text, stamp, logo), return: {"is_signature": false}"""

        try:
            response = self.provider.multimodal_query(sig_bytes, validation_prompt)
            
            # Parse JSON response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                return None
            
            # Check if it's actually a signature
            if not data.get('is_signature', False):
                return None
            
            sig_block = SignatureBlock(
                name=data.get('name'),
                designation=data.get('designation'),
                date=data.get('date'),
                visual_description=data.get('visual_description', ''),
                page_number=page.page_number,
                bbox=bbox,
                image=self._crop_region(page.image, bbox, padding=10),  # Tighter crop for storage
                confidence=0.85  # CV+LLM validated
            )
            
            # Save signature image if configured (thread-safe)
            if self.config.save_signatures and self.config.output_dir:
                with self._signature_lock:
                    sig_block.image_path = self._save_signature_image(
                        sig_block.image if sig_block.image is not None else sig_image,
                        page.page_number,
                        self._signature_counter
                    )
                    self._signature_counter += 1
            
            return sig_block
            
        except Exception:
            return None
    
    def _overlaps_with_any(
        self,
        bbox: BoundingBox,
        existing: List[BoundingBox],
        threshold: float = 0.5
    ) -> bool:
        """
        Check if a bounding box overlaps significantly with any existing boxes.
        
        Args:
            bbox: Bounding box to check
            existing: List of existing bounding boxes
            threshold: IoU threshold for overlap (0-1)
            
        Returns:
            True if bbox overlaps with any existing box above threshold
        """
        for other in existing:
            iou = self._compute_iou(bbox, other)
            if iou > threshold:
                return True
        return False
    
    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score (0-1)
        """
        # Get coordinates
        x1_1, y1_1 = box1.x, box1.y
        x2_1, y2_1 = box1.x + box1.width, box1.y + box1.height
        
        x1_2, y1_2 = box2.x, box2.y
        x2_2, y2_2 = box2.x + box2.width, box2.y + box2.height
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_signature_region(
        self,
        page: PageImage,
        region: Region,
        layout: LayoutResult
    ) -> Optional[SignatureBlock]:
        """
        Analyze a detected signature region.
        
        Args:
            page: Page image
            region: Signature region from layout
            
        Returns:
            SignatureBlock or None
        """
        # Crop signature region with context
        bbox = self._bbox_to_pixels(region.bbox, layout, page) if region.bbox else None
        sig_image = self._crop_region(
            page.image,
            bbox,
            padding=self.config.signature_context_padding
        )
        
        if sig_image is None:
            return None
        
        sig_bytes = self._image_to_bytes(sig_image)
        
        # Analyze with LLM
        try:
            sig_block = self.provider.analyze_signature(sig_bytes)
            sig_block.page_number = page.page_number
            sig_block.bbox = bbox or region.bbox
            sig_block.image = sig_image
            
            # Save signature image if configured (thread-safe)
            if self.config.save_signatures and self.config.output_dir:
                with self._signature_lock:
                    sig_block.image_path = self._save_signature_image(
                        sig_image,
                        page.page_number,
                        self._signature_counter
                    )
                    self._signature_counter += 1
            
            return sig_block
        except Exception:
            return None
    
    def _detect_signatures_with_llm(
        self,
        page: PageImage,
        image_bytes: bytes
    ) -> List[SignatureBlock]:
        """
        Use LLM to detect and analyze signatures in the page.
        
        Args:
            page: Page image
            image_bytes: Page image bytes
            
        Returns:
            List of SignatureBlock objects
        """
        detection_prompt = """Analyze this document page for handwritten signatures.

Return a JSON array. Each item must include:
- bbox: [x1, y1, x2, y2] as fractions of page width/height (0 to 1)
- name: printed name near the signature (or null)
- designation: title near the signature (or null)
- date: any date near the signature (or null)
- visual_description: brief visual description

Rules:
- Only include actual handwritten signatures (exclude logos, stamps, printed names).
- If unsure, do NOT include the region.
- If no signatures, return [].

Example:
[{"bbox":[0.62,0.72,0.92,0.80],"name":"John Smith","designation":"Managing Director","date":"15/01/2026","visual_description":"cursive with loop"}]"""

        try:
            response = self.provider.multimodal_query(image_bytes, detection_prompt)
            
            # Parse JSON response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                sig_data = json.loads(response[json_start:json_end])
            else:
                return []
            
            signatures = []
            for data in sig_data:
                bbox_data = data.get("bbox") or []
                if len(bbox_data) != 4:
                    continue
                x1, y1, x2, y2 = bbox_data
                x1 = max(0.0, min(1.0, float(x1)))
                y1 = max(0.0, min(1.0, float(y1)))
                x2 = max(0.0, min(1.0, float(x2)))
                y2 = max(0.0, min(1.0, float(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                
                bbox = BoundingBox(
                    x=x1 * page.width,
                    y=y1 * page.height,
                    width=(x2 - x1) * page.width,
                    height=(y2 - y1) * page.height
                )
                
                sig_image = self._crop_region(
                    page.image,
                    bbox,
                    padding=self.config.signature_context_padding
                )
                
                sig = SignatureBlock(
                    name=data.get('name'),
                    designation=data.get('designation'),
                    date=data.get('date'),
                    visual_description=data.get('visual_description', ''),
                    page_number=page.page_number,
                    bbox=bbox,
                    image=sig_image,
                    confidence=0.75  # LLM-only detection confidence
                )
                
                if self.config.save_signatures and self.config.output_dir and sig_image is not None:
                    with self._signature_lock:
                        sig.image_path = self._save_signature_image(
                            sig_image,
                            page.page_number,
                            self._signature_counter
                        )
                        self._signature_counter += 1
                
                signatures.append(sig)
            
            return signatures
        except Exception:
            return []
    
    def _detect_redactions(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect redacted (blacked-out) regions in an image.
        
        Args:
            image: Image as numpy array (BGR)
            
        Returns:
            List of bounding boxes for redacted regions
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold for very dark regions
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        redactions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter criteria for redactions:
            # - Minimum area (not tiny spots)
            # - Reasonable aspect ratio (not extremely thin lines)
            # - Rectangular shape (high solidity)
            if area > 500 and 0.1 < aspect_ratio < 15:
                # Check if region is mostly black
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    mean_val = np.mean(roi)
                    # Check solidity (how rectangular it is)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    if mean_val < 50 and solidity > 0.7:
                        redactions.append(BoundingBox(
                            x=float(x),
                            y=float(y),
                            width=float(w),
                            height=float(h)
                        ))
        
        return redactions
    
    def _crop_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: int = 0
    ) -> Optional[np.ndarray]:
        """
        Crop a region from an image.
        
        Args:
            image: Source image
            bbox: Bounding box to crop
            padding: Pixels to add around the region
            
        Returns:
            Cropped image or None if invalid
        """
        if bbox is None:
            return None
        
        height, width = image.shape[:2]
        
        # Calculate crop coordinates with padding
        x1 = max(0, int(bbox.x) - padding)
        y1 = max(0, int(bbox.y) - padding)
        x2 = min(width, int(bbox.x + bbox.width) + padding)
        y2 = min(height, int(bbox.y + bbox.height) + padding)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2].copy()

    def _bbox_to_pixels(
        self,
        bbox: Optional[BoundingBox],
        layout: Optional[LayoutResult],
        page: PageImage
    ) -> Optional[BoundingBox]:
        """Convert layout bbox to pixel coordinates if needed."""
        if bbox is None:
            return None
        if layout is None or layout.width <= 0 or layout.height <= 0:
            return bbox
        
        scale_x = page.width / layout.width
        scale_y = page.height / layout.height
        
        # If scale is ~1, bbox is already in pixels
        if 0.9 <= scale_x <= 1.1 and 0.9 <= scale_y <= 1.1:
            return bbox
        
        return BoundingBox(
            x=bbox.x * scale_x,
            y=bbox.y * scale_y,
            width=bbox.width * scale_x,
            height=bbox.height * scale_y
        )
    
    def _image_to_bytes(self, image: np.ndarray, format: str = "png") -> bytes:
        """Convert numpy image to bytes."""
        import cv2
        
        if format.lower() == "png":
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return buffer.tobytes()
    
    def _extract_markdown_table(self, response: str) -> str:
        """Extract Markdown table from LLM response."""
        lines = response.strip().split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|'):
                in_table = True
                table_lines.append(line)
            elif in_table and stripped and not stripped.startswith('|'):
                # End of table
                break
            elif in_table and not stripped:
                # Empty line might be end of table
                continue
        
        return '\n'.join(table_lines) if table_lines else ""

    def _markdown_table_matches_dimensions(
        self,
        markdown: str,
        expected_rows: int,
        expected_cols: int
    ) -> bool:
        """Validate markdown table size against expected dimensions."""
        if not markdown:
            return False
        
        lines = [line.strip() for line in markdown.strip().split('\n') if line.strip()]
        rows = []
        for line in lines:
            if not line.startswith('|'):
                continue
            # Skip separator rows
            if set(line.replace('|', '').strip()) <= {'-', ':'}:
                continue
            parts = [p.strip() for p in line.split('|')]
            if parts and parts[0] == '':
                parts = parts[1:]
            if parts and parts[-1] == '':
                parts = parts[:-1]
            if parts:
                rows.append(parts)
        
        if not rows:
            return False
        
        col_count = max(len(r) for r in rows)
        row_count = len(rows)
        
        return row_count == expected_rows and col_count == expected_cols

    def _dedupe_regions(
        self,
        regions: List[BoundingBox],
        threshold: float = 0.3
    ) -> List[BoundingBox]:
        """Remove overlapping regions using IoU threshold."""
        deduped = []
        for bbox in regions:
            if self._overlaps_with_any(bbox, deduped, threshold=threshold):
                continue
            deduped.append(bbox)
        return deduped
    
    def _save_signature_image(
        self,
        image: np.ndarray,
        page_number: int,
        sig_index: int
    ) -> str:
        """Save signature image to file."""
        import cv2
        
        output_dir = Path(self.config.output_dir) / "signatures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"sig_page{page_number:03d}_{sig_index:02d}.png"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), image)
        
        return str(filepath)
