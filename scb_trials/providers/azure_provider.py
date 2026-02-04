"""
Azure Provider Implementation
Uses Azure Document Intelligence + Azure OpenAI GPT-5.2 for document extraction.
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


class AzureProvider(OCRProvider):
    """
    Azure-based OCR provider using:
    - Azure Document Intelligence (Form Recognizer) for layout and structure
    - Azure OpenAI GPT-5.2 for multimodal understanding
    
    Falls back to LLM-only mode if Document Intelligence is not configured.
    """
    
    def __init__(self):
        """Initialize Azure clients."""
        super().__init__(name="azure")
        
        # Document Intelligence credentials
        self.di_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.di_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        # OpenAI credentials
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        # Initialize clients lazily
        self._di_client = None
        self._openai_client = None
        
        # Require Document Intelligence and Azure OpenAI credentials
        if not self.di_endpoint or not self.di_key:
            raise ValueError(
                "Azure Document Intelligence credentials not found. "
                "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY"
            )
        if not self.openai_endpoint or not self.openai_key or not self.openai_deployment:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT_NAME"
            )
        
        self.llm_only_mode = False
    
    @property
    def di_client(self):
        """Lazy initialization of Document Intelligence client."""
        if self._di_client is None:
            if not self.di_endpoint or not self.di_key:
                raise ValueError(
                    "Azure Document Intelligence credentials not found. "
                    "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY"
                )
            
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            self._di_client = DocumentAnalysisClient(
                endpoint=self.di_endpoint,
                credential=AzureKeyCredential(self.di_key)
            )
        return self._di_client
    
    @property
    def openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            if not self.openai_endpoint or not self.openai_key:
                raise ValueError(
                    "Azure OpenAI credentials not found. "
                    "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
                )
            
            from openai import AzureOpenAI
            
            self._openai_client = AzureOpenAI(
                azure_endpoint=self.openai_endpoint,
                api_key=self.openai_key,
                api_version=self.openai_api_version
            )
        return self._openai_client
    
    def analyze_layout(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze document layout.
        
        If Document Intelligence is configured, uses prebuilt-layout model.
        Otherwise, falls back to LLM-only analysis.
        """
        return self._analyze_layout_with_di(image, page_number)
    
    def _analyze_layout_llm_only(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze layout using only the multimodal LLM.
        
        This is a fallback when Document Intelligence is not available.
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
    
    def _analyze_layout_with_di(self, image: bytes, page_number: int = 1) -> LayoutResult:
        """
        Analyze document layout using Azure Document Intelligence.
        
        Uses the prebuilt-layout model which detects:
        - Paragraphs and text blocks
        - Tables with cell coordinates
        - Selection marks
        - Document structure
        """
        # Analyze with Document Intelligence
        poller = self.di_client.begin_analyze_document(
            "prebuilt-layout",
            document=image
        )
        result = poller.result()
        
        regions = []
        raw_text = result.content if result.content else ""
        
        # Get page dimensions
        page = result.pages[0] if result.pages else None
        width = int(page.width) if page else 0
        height = int(page.height) if page else 0
        
        # Process paragraphs as text blocks
        if result.paragraphs:
            for para in result.paragraphs:
                if para.bounding_regions:
                    br = para.bounding_regions[0]
                    # Convert polygon to bounding box
                    bbox = self._polygon_to_bbox(br.polygon) if hasattr(br, 'polygon') else None
                    
                    regions.append(Region(
                        region_type=RegionType.TEXT_BLOCK,
                        bbox=bbox or BoundingBox(0, 0, 0, 0),
                        content=para.content,
                        confidence=1.0
                    ))
        
        # Process tables
        if result.tables:
            for table in result.tables:
                cells = []
                for cell in table.cells:
                    cells.append(TableCell(
                        row_index=cell.row_index,
                        column_index=cell.column_index,
                        content=cell.content,
                        row_span=cell.row_span or 1,
                        column_span=cell.column_span or 1,
                        confidence=1.0
                    ))
                
                table_obj = Table(
                    cells=cells,
                    row_count=table.row_count,
                    column_count=table.column_count,
                    confidence=1.0
                )
                
                # Get table bounding box
                bbox = None
                if table.bounding_regions:
                    br = table.bounding_regions[0]
                    bbox = self._polygon_to_bbox(br.polygon) if hasattr(br, 'polygon') else None
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
        if page and hasattr(page, 'words'):
            confidences = [w.confidence for w in page.words if hasattr(w, 'confidence')]
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
        """Extract text using Azure Document Intelligence or LLM fallback."""
        if self.llm_only_mode:
            prompt = """Extract ALL text from this document image exactly as written.
Preserve:
- All formatting and structure
- Line breaks and paragraph structure
- Special characters and symbols
- Financial values character-for-character exact

Output the text only, no explanation."""
            return self.multimodal_query(image, prompt)
        
        poller = self.di_client.begin_analyze_document(
            "prebuilt-read",
            document=image
        )
        result = poller.result()
        return result.content if result.content else ""
    
    def extract_tables(self, image: bytes) -> List[Table]:
        """Extract tables using Document Intelligence or LLM."""
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
        Send a multimodal query to Azure OpenAI GPT-5.2.
        
        Args:
            image: Image bytes (PNG/JPEG)
            prompt: Text prompt for the model
            
        Returns:
            Model response as string
        """
        # Encode image to base64
        image_base64 = base64.b64encode(image).decode('utf-8')
        
        # Determine image type (assume PNG if not specified)
        image_type = "image/png"
        if image[:3] == b'\xff\xd8\xff':
            image_type = "image/jpeg"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # GPT-5.x models use max_completion_tokens instead of max_tokens
        response = self.openai_client.chat.completions.create(
            model=self.openai_deployment,
            messages=messages,
            max_completion_tokens=4096,
            temperature=0.1  # Low temperature for accuracy
        )
        
        return response.choices[0].message.content
    
    def analyze_signature(self, image: bytes, context_image: Optional[bytes] = None) -> SignatureBlock:
        """
        Analyze a signature region using Azure OpenAI.
        
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
        Use LLM to generate accurate Markdown representation of a table.
        
        Args:
            image: Full page image
            table: Table object with cell data from Document Intelligence
            
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
    
    def _polygon_to_bbox(self, polygon) -> Optional[BoundingBox]:
        """Convert a polygon to a bounding box."""
        if not polygon:
            return None
        
        try:
            # Polygon is a list of points
            if hasattr(polygon[0], 'x'):
                xs = [p.x for p in polygon]
                ys = [p.y for p in polygon]
            else:
                # Flat list of coordinates [x1, y1, x2, y2, ...]
                xs = polygon[0::2]
                ys = polygon[1::2]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            return BoundingBox(
                x=min_x,
                y=min_y,
                width=max_x - min_x,
                height=max_y - min_y
            )
        except (IndexError, TypeError):
            return None
    
    def detect_redactions(self, image: bytes) -> List[BoundingBox]:
        """
        Detect redacted (blacked-out) regions in an image.
        Uses OpenCV for detection, not Document Intelligence.
        
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
            # Redactions are typically rectangular with reasonable aspect ratio
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
