"""
Stage 2: Extractor
Handles OCR and content extraction using providers.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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
    signature_aspect_ratio_min: float = 1.0  # Signatures are usually wider than tall
    signature_aspect_ratio_max: float = 20.0
    signature_page_region: str = "lower_half"  # Default region if search list not provided
    signature_search_regions: List[str] = field(default_factory=lambda: ["lower_half", "full"])
    signature_ink_density_min: float = 0.01  # Minimum ink density (not too sparse)
    signature_ink_density_max: float = 0.85  # Maximum ink density (avoid solid blocks)
    signature_fill_ratio_min: float = 0.1
    signature_fill_ratio_max: float = 0.98
    signature_variance_min: float = 50.0


class Extractor:
    """
    Stage 2: Extractor
    
    Handles:
    1. Layout analysis using document intelligence services
    2. Region-specific OCR (text blocks, tables, signatures)
    3. Redaction detection
    4. LLM-enhanced extraction for accuracy
    """
    
    def __init__(self, provider: OCRProvider, config: Optional[ExtractorConfig] = None):
        """
        Initialize extractor with a provider.
        
        Args:
            provider: OCR provider (Azure or Vertex)
            config: Extraction configuration
        """
        self.provider = provider
        self.config = config or ExtractorConfig()
        self._signature_counter = 0
    
    def extract_document(
        self,
        pages: List[PageImage],
        source_file: str
    ) -> ExtractionResult:
        """
        Extract content from all pages of a document.
        
        Args:
            pages: List of preprocessed page images
            source_file: Original source file path
            
        Returns:
            ExtractionResult with all extracted content
        """
        page_extractions = []
        
        for page in pages:
            extraction = self.extract_page(page)
            page_extractions.append(extraction)
        
        # Calculate overall confidence
        confidences = [p.confidence for p in page_extractions]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExtractionResult(
            source_file=source_file,
            provider=self.provider.name,
            pages=page_extractions,
            total_pages=len(pages),
            overall_confidence=overall_confidence
        )
    
    def extract_page(self, page: PageImage) -> PageExtraction:
        """
        Extract content from a single page.
        
        Args:
            page: Preprocessed page image
            
        Returns:
            PageExtraction with all extracted content
        """
        # Get image bytes for provider
        image_bytes = page.to_bytes()
        
        # Step 1: Analyze layout to get regions
        layout = self.provider.analyze_layout(image_bytes, page.page_number)
        
        # Step 2: Detect redactions (if configured)
        redactions = []
        if self.config.detect_redactions:
            redactions = self._detect_redactions(page.image)
        
        # Step 3: Detect signatures (may not be in layout)
        signatures = []
        if self.config.extract_signatures:
            signatures = self._extract_signatures(page, layout, image_bytes)
        
        # Step 4: Process tables
        tables = []
        if self.config.extract_tables:
            tables = self._extract_tables(page, layout, image_bytes)
        
        # Step 5: Extract text blocks
        text_blocks = self._extract_text_blocks(layout, image_bytes)
        
        # Calculate page confidence
        confidence = layout.confidence
        
        return PageExtraction(
            page_number=page.page_number,
            layout=layout,
            text_blocks=text_blocks,
            tables=tables,
            signatures=signatures,
            redactions=redactions,
            confidence=confidence
        )
    
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
            
            # Save signature image if configured
            if self.config.save_signatures and self.config.output_dir:
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
            
            # Save signature image if configured
            if self.config.save_signatures and self.config.output_dir:
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
