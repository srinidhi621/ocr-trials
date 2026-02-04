"""
Vertex Provider Implementation
Uses Google Document AI + Gemini 3 Pro for document extraction.
"""

import os
import base64
import json
from typing import List, Optional
from dotenv import load_dotenv

from .base import (
    OCRProvider,
    LayoutResult,
    RegionType,
    Region,
    Table,
    TableCell,
    SignatureBlock,
    BoundingBox,
)

# Load environment variables
load_dotenv()


class VertexProvider(OCRProvider):
    """
    Google Cloud-based OCR provider using:
    - Google Document AI for layout and structure
    - Gemini 3 Pro for multimodal understanding
    
    Falls back to LLM-only mode if Document AI is not configured.
    """
    
    def __init__(self):
        """Initialize Google Cloud clients."""
        super().__init__(name="vertex")
        
        # Document AI configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us")
        self.processor_id = os.getenv("GOOGLE_DOCUMENT_AI_PROCESSOR_ID")
        
        # Gemini configuration
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
        
        # Initialize clients lazily
        self._docai_client = None
        self._gemini_client = None
        
        # Check if we're in LLM-only mode
        self.llm_only_mode = not (self.project_id and self.processor_id)
        if self.llm_only_mode:
            print("  [Note] Document AI not configured, using LLM-only mode with Gemini")
    
    @property
    def docai_client(self):
        """Lazy initialization of Document AI client."""
        if self._docai_client is None:
            if not self.project_id or not self.processor_id:
                raise ValueError(
                    "Google Document AI credentials not found. "
                    "Set GOOGLE_CLOUD_PROJECT and GOOGLE_DOCUMENT_AI_PROCESSOR_ID"
                )
            
            from google.cloud import documentai
            
            self._docai_client = documentai.DocumentProcessorServiceClient()
        return self._docai_client
    
    @property
    def gemini_client(self):
        """Lazy initialization of Gemini client."""
        if self._gemini_client is None:
            if not self.api_key:
                raise ValueError(
                    "Google API key not found. Set GOOGLE_API_KEY"
                )
            
            from google import genai
            
            self._gemini_client = genai.Client(api_key=self.api_key)
        return self._gemini_client
    
    @property
    def processor_name(self) -> str:
        """Get the full processor resource name."""
        return (
            f"projects/{self.project_id}/locations/{self.location}"
            f"/processors/{self.processor_id}"
        )
    
    def analyze_layout(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze document layout.
        
        If Document AI is configured, uses the Document AI processor.
        Otherwise, falls back to LLM-only analysis using Gemini.
        """
        if self.llm_only_mode:
            return self._analyze_layout_llm_only(image, page_number)
        
        return self._analyze_layout_with_docai(image, page_number)
    
    def _analyze_layout_llm_only(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze layout using only Gemini multimodal LLM.
        
        This is a fallback when Document AI is not available.
        Uses a multi-step approach for better reliability.
        """
        regions = []
        raw_text = ""
        
        # Step 1: Extract all text directly (most reliable)
        text_prompt = """Extract ALL text from this document page exactly as written.

IMPORTANT RULES:
1. Extract EVERY piece of text you can see
2. Preserve the exact wording - do not paraphrase or summarize
3. Preserve line breaks and paragraph structure
4. Include headers, footers, page numbers, everything
5. For tables, extract the content row by row
6. Financial values must be character-for-character exact (e.g., "INR 500,000,000.00")

Output the text content only, no explanations or formatting instructions."""

        try:
            raw_text = self.multimodal_query(image, text_prompt)
            
            if raw_text and raw_text.strip():
                # Split into paragraphs
                paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
                
                # If no double-newline splits, try single newlines for very structured content
                if len(paragraphs) <= 1 and '\n' in raw_text:
                    paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]
                
                for i, para in enumerate(paragraphs):
                    if para:
                        regions.append(Region(
                            region_type=RegionType.TEXT_BLOCK,
                            bbox=BoundingBox(0, float(i * 50), 100, 40),
                            content=para,
                            confidence=0.9
                        ))
        except Exception as e:
            print(f"    [Warning] Text extraction failed for page {page_number}: {e}")
        
        # Step 2: Check for tables
        if raw_text:
            table_prompt = """Look at this document page. Are there any tables present?

If YES, extract each table in Markdown format with | delimiters.
If NO, respond with: NO_TABLES

For tables:
- Preserve exact structure (rows/columns)
- Copy all values verbatim
- Include header row and separator

Output only the Markdown table(s) or NO_TABLES."""

            try:
                table_response = self.multimodal_query(image, table_prompt)
                
                if table_response and "NO_TABLES" not in table_response.upper():
                    # Parse markdown tables
                    tables = self._parse_markdown_tables(table_response)
                    for i, table in enumerate(tables):
                        table.bbox = BoundingBox(0, float(300 + i * 150), 100, 100)
                        regions.append(Region(
                            region_type=RegionType.TABLE,
                            bbox=table.bbox,
                            content="",
                            table=table,
                            confidence=0.85
                        ))
            except Exception as e:
                print(f"    [Warning] Table extraction failed for page {page_number}: {e}")
        
        # Step 3: Check for signatures
        sig_prompt = """Look at this document page. Are there any signatures present?

If YES, provide details in this exact format:
SIGNATURE_FOUND
Name: [printed name if visible, or UNKNOWN]
Designation: [title if visible, or UNKNOWN]
Description: [brief visual description of signature]

If NO signatures, respond with: NO_SIGNATURES"""

        try:
            sig_response = self.multimodal_query(image, sig_prompt)
            
            if sig_response and "SIGNATURE_FOUND" in sig_response.upper():
                # Parse signature info
                sig_block = self._parse_signature_response(sig_response, page_number)
                if sig_block:
                    regions.append(Region(
                        region_type=RegionType.SIGNATURE,
                        bbox=BoundingBox(0, 700, 100, 50),
                        content="",
                        signature=sig_block,
                        confidence=0.8
                    ))
        except Exception as e:
            print(f"    [Warning] Signature detection failed for page {page_number}: {e}")
        
        return LayoutResult(
            page_number=page_number,
            regions=regions,
            width=0,
            height=0,
            raw_text=raw_text,
            confidence=0.85 if regions else 0.5
        )
    
    def _parse_markdown_tables(self, response: str) -> List[Table]:
        """Parse markdown tables from LLM response."""
        tables = []
        lines = response.strip().split('\n')
        
        current_table_lines = []
        in_table = False
        
        for line in lines:
            if line.strip().startswith('|'):
                in_table = True
                current_table_lines.append(line)
            elif in_table and (not line.strip() or not line.strip().startswith('|')):
                # End of table
                if current_table_lines:
                    table = self._lines_to_table(current_table_lines)
                    if table:
                        tables.append(table)
                current_table_lines = []
                in_table = False
        
        # Don't forget the last table
        if current_table_lines:
            table = self._lines_to_table(current_table_lines)
            if table:
                tables.append(table)
        
        return tables
    
    def _lines_to_table(self, lines: List[str]) -> Optional[Table]:
        """Convert markdown table lines to Table object."""
        cells = []
        rows = []
        
        for line in lines:
            # Skip separator lines (|---|---|)
            if '---' in line:
                continue
            
            # Parse cells
            parts = [p.strip() for p in line.split('|')]
            # Remove empty strings from start and end
            parts = [p for p in parts if p or parts.index(p) not in [0, len(parts)-1]]
            if not parts:
                continue
            
            rows.append(parts)
        
        if not rows:
            return None
        
        # Convert to cells
        for row_idx, row in enumerate(rows):
            for col_idx, content in enumerate(row):
                cells.append(TableCell(
                    row_index=row_idx,
                    column_index=col_idx,
                    content=content,
                    confidence=0.85
                ))
        
        col_count = max(len(row) for row in rows) if rows else 0
        
        table = Table(
            cells=cells,
            row_count=len(rows),
            column_count=col_count,
            confidence=0.85
        )
        table.markdown = '\n'.join(lines)
        
        return table
    
    def _parse_signature_response(self, response: str, page_number: int) -> Optional[SignatureBlock]:
        """Parse signature information from LLM response."""
        name = None
        designation = None
        description = ""
        
        for line in response.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith('name:'):
                value = line.split(':', 1)[1].strip()
                if value.upper() != 'UNKNOWN':
                    name = value
            elif line_lower.startswith('designation:'):
                value = line.split(':', 1)[1].strip()
                if value.upper() != 'UNKNOWN':
                    designation = value
            elif line_lower.startswith('description:'):
                description = line.split(':', 1)[1].strip()
        
        return SignatureBlock(
            name=name,
            designation=designation,
            page_number=page_number,
            visual_description=description,
            confidence=0.8
        )
    
    def _analyze_layout_with_docai(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze document layout using Google Document AI.
        
        Uses the Document AI processor which detects:
        - Text blocks and paragraphs
        - Tables with cell coordinates
        - Form fields
        - Document structure
        """
        from google.cloud import documentai
        
        # Create the request
        raw_document = documentai.RawDocument(
            content=image,
            mime_type="image/png"
        )
        
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        
        # Process the document
        result = self.docai_client.process_document(request=request)
        document = result.document
        
        regions = []
        raw_text = document.text if document.text else ""
        
        # Get page dimensions
        page = document.pages[0] if document.pages else None
        width = int(page.dimension.width) if page and page.dimension else 0
        height = int(page.dimension.height) if page and page.dimension else 0
        
        # Process text blocks (paragraphs)
        if page and page.paragraphs:
            for para in page.paragraphs:
                bbox = self._layout_to_bbox(para.layout, width, height)
                text = self._get_text_from_layout(para.layout, document.text)
                
                regions.append(Region(
                    region_type=RegionType.TEXT_BLOCK,
                    bbox=bbox or BoundingBox(0, 0, 0, 0),
                    content=text,
                    confidence=para.layout.confidence if para.layout else 1.0
                ))
        
        # Process tables
        if page and page.tables:
            for table in page.tables:
                cells = []
                
                # Process header rows
                for row_idx, row in enumerate(table.header_rows):
                    for cell_idx, cell in enumerate(row.cells):
                        cell_text = self._get_text_from_layout(cell.layout, document.text)
                        cells.append(TableCell(
                            row_index=row_idx,
                            column_index=cell_idx,
                            content=cell_text,
                            row_span=cell.row_span or 1,
                            column_span=cell.col_span or 1,
                            confidence=cell.layout.confidence if cell.layout else 1.0
                        ))
                
                # Process body rows
                header_row_count = len(table.header_rows)
                for row_idx, row in enumerate(table.body_rows):
                    for cell_idx, cell in enumerate(row.cells):
                        cell_text = self._get_text_from_layout(cell.layout, document.text)
                        cells.append(TableCell(
                            row_index=header_row_count + row_idx,
                            column_index=cell_idx,
                            content=cell_text,
                            row_span=cell.row_span or 1,
                            column_span=cell.col_span or 1,
                            confidence=cell.layout.confidence if cell.layout else 1.0
                        ))
                
                # Determine table dimensions
                row_count = len(table.header_rows) + len(table.body_rows)
                col_count = max(
                    (len(row.cells) for row in table.header_rows),
                    default=0
                )
                if table.body_rows:
                    col_count = max(col_count, max(len(row.cells) for row in table.body_rows))
                
                table_obj = Table(
                    cells=cells,
                    row_count=row_count,
                    column_count=col_count,
                    confidence=1.0
                )
                
                bbox = self._layout_to_bbox(table.layout, width, height) if table.layout else None
                table_obj.bbox = bbox
                
                regions.append(Region(
                    region_type=RegionType.TABLE,
                    bbox=bbox or BoundingBox(0, 0, 0, 0),
                    content="",
                    table=table_obj,
                    confidence=1.0
                ))
        
        # Calculate overall confidence
        confidence = 1.0
        if page and page.blocks:
            confidences = [b.layout.confidence for b in page.blocks if b.layout and b.layout.confidence]
            if confidences:
                confidence = sum(confidences) / len(confidences)
        
        return LayoutResult(
            page_number=page_number,
            regions=regions,
            width=width,
            height=height,
            raw_text=raw_text,
            confidence=confidence
        )
    
    def extract_text(self, image: bytes) -> str:
        """Extract text using Google Document AI or LLM fallback."""
        if self.llm_only_mode:
            prompt = """Extract ALL text from this document image exactly as written.
Preserve:
- All formatting and structure
- Line breaks and paragraph structure
- Special characters and symbols
- Financial values character-for-character exact

Output the text only, no explanation."""
            return self.multimodal_query(image, prompt)
        
        layout = self.analyze_layout(image)
        return layout.raw_text
    
    def extract_tables(self, image: bytes) -> List[Table]:
        """Extract tables using Google Document AI or LLM."""
        layout = self.analyze_layout(image)
        tables = []
        
        for region in layout.regions:
            if region.region_type == RegionType.TABLE and region.table:
                table = region.table
                # In LLM-only mode, get better Markdown directly
                if self.llm_only_mode:
                    table.markdown = self._extract_table_markdown_llm(image)
                else:
                    table.markdown = self._enhance_table_with_llm(image, table)
                tables.append(table)
        
        return tables
    
    def _extract_table_markdown_llm(self, image: bytes) -> str:
        """Extract table as Markdown using only LLM."""
        prompt = """Extract any tables from this image and convert them to Markdown format.

STRICT RULES:
1. Preserve EXACT structure (rows/columns)
2. Copy ALL values VERBATIM - no summarization
3. Financial values must be character-for-character exact
4. If multiple tables exist, separate them with a blank line

Output ONLY the Markdown table(s), no explanation."""
        
        response = self.multimodal_query(image, prompt)
        
        # Extract just the markdown tables
        lines = response.strip().split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if line.strip().startswith('|'):
                in_table = True
                table_lines.append(line)
            elif in_table and line.strip() and not line.strip().startswith('|'):
                table_lines.append('')  # Separator between tables
                in_table = False
        
        return '\n'.join(table_lines)
    
    def multimodal_query(self, image: bytes, prompt: str) -> str:
        """
        Send a multimodal query to Gemini 3 Pro.
        
        Args:
            image: Image bytes (PNG/JPEG)
            prompt: Text prompt for the model
            
        Returns:
            Model response as string
        """
        from google.genai import types
        import PIL.Image
        import io
        
        # Convert bytes to PIL Image
        pil_image = PIL.Image.open(io.BytesIO(image))
        
        # Create generation config
        config = types.GenerateContentConfig(
            temperature=0.1  # Low temperature for accuracy
        )
        
        # Send request to Gemini
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=[prompt, pil_image],
            config=config
        )
        
        return response.text
    
    def analyze_signature(self, image: bytes, context_image: Optional[bytes] = None) -> SignatureBlock:
        """
        Analyze a signature region using Gemini.
        
        Args:
            image: Cropped signature image
            context_image: Optional wider context with name/designation
        """
        prompt = """Analyze this signature block. Extract the following information:

1. **Name**: The printed name associated with the signature (if visible)
2. **Designation**: The job title or role (e.g., "Managing Director", "CFO")
3. **Date**: Any date visible near the signature
4. **Visual Description**: Brief description of the signature's visual characteristics (e.g., "cursive with loop", "initials only")

Return the information as JSON with keys: name, designation, date, visual_description

If any field is not visible or cannot be determined, use null for that field."""

        # Use context image if provided (contains more area around signature)
        query_image = context_image if context_image else image
        
        response = self.multimodal_query(query_image, prompt)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}
        
        return SignatureBlock(
            name=data.get('name'),
            designation=data.get('designation'),
            date=data.get('date'),
            visual_description=data.get('visual_description', ''),
            confidence=0.9  # Default confidence for LLM extraction
        )
    
    def _enhance_table_with_llm(self, image: bytes, table: Table) -> str:
        """
        Use Gemini to generate accurate Markdown representation of a table.
        
        Args:
            image: Full page image
            table: Table object with cell data from Document AI
            
        Returns:
            Markdown representation of the table
        """
        # Convert table to grid for context
        grid = table.to_grid()
        table_json = json.dumps(grid, indent=2)
        
        prompt = f"""Convert this table to Markdown format.

STRICT RULES:
1. Preserve EXACT structure (rows/columns) as shown
2. For merged cells, repeat the content or use appropriate representation
3. Copy ALL values VERBATIM - do not summarize or modify any values
4. Financial values must be character-for-character exact (e.g., "INR 500,000,000.00")
5. Do not add any extra rows or columns

Table data from OCR:
{table_json}

The original table image is attached for verification. Output ONLY the Markdown table, no explanation."""

        response = self.multimodal_query(image, prompt)
        
        # Extract just the markdown table from response
        lines = response.strip().split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if line.strip().startswith('|'):
                in_table = True
                table_lines.append(line)
            elif in_table and not line.strip().startswith('|'):
                break
        
        return '\n'.join(table_lines) if table_lines else table.to_markdown()
    
    def _layout_to_bbox(self, layout, page_width: int, page_height: int) -> Optional[BoundingBox]:
        """Convert Document AI layout to bounding box."""
        if not layout or not layout.bounding_poly:
            return None
        
        try:
            vertices = layout.bounding_poly.normalized_vertices
            if not vertices or len(vertices) < 4:
                return None
            
            # Normalized coordinates (0-1), convert to absolute
            xs = [v.x * page_width for v in vertices]
            ys = [v.y * page_height for v in vertices]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            return BoundingBox(
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y
            )
        except (AttributeError, IndexError):
            return None
    
    def _get_text_from_layout(self, layout, full_text: str) -> str:
        """Extract text for a layout element using text anchors."""
        if not layout or not layout.text_anchor or not layout.text_anchor.text_segments:
            return ""
        
        text_parts = []
        for segment in layout.text_anchor.text_segments:
            start = int(segment.start_index) if segment.start_index else 0
            end = int(segment.end_index) if segment.end_index else len(full_text)
            text_parts.append(full_text[start:end])
        
        return "".join(text_parts).strip()
    
    def detect_redactions(self, image: bytes) -> List[BoundingBox]:
        """
        Detect redacted (blacked-out) regions in an image.
        Uses OpenCV for detection.
        
        Args:
            image: Image bytes
            
        Returns:
            List of bounding boxes for redacted regions
        """
        import cv2
        import numpy as np
        
        # Decode image
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold for dark regions
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        redactions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = cv2.contourArea(contour)
            
            # Filter for rectangular shapes that look like redactions
            if 0.2 < aspect_ratio < 10 and area > 500:
                # Check if the region is mostly black
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    mean_val = np.mean(roi)
                    if mean_val < 50:  # Dark region
                        redactions.append(BoundingBox(
                            x=float(x),
                            y=float(y),
                            width=float(w),
                            height=float(h)
                        ))
        
        return redactions
