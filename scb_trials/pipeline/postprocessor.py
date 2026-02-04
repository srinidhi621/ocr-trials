"""
Stage 3: Postprocessor
Handles content merging, signature comparison, and financial validation.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from providers.base import (
    ExtractionResult,
    PageExtraction,
    SignatureBlock,
    Table,
    BoundingBox,
    RegionType,
)


@dataclass
class FinancialValue:
    """Represents an extracted financial value."""
    value: str  # Original string as extracted
    page: int
    context: str = ""  # Surrounding text for context
    value_type: str = ""  # currency, percentage, amount
    confidence: float = 1.0


@dataclass
class SignatureComparison:
    """Result of comparing signatures for the same person."""
    name: str
    designation: Optional[str]
    pages: List[int]
    comparison_result: str  # "Consistent", "Discrepancy", "Single occurrence"
    similarity_scores: List[float] = field(default_factory=list)
    notes: str = ""


@dataclass
class ContentBlock:
    """A block of content in the merged document."""
    block_type: str  # "text", "table", "signature", "redacted"
    content: str  # Text content or Markdown
    page: int
    position: Tuple[float, float]  # (y, x) for sorting
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Complete processed document after post-processing."""
    source_file: str
    provider: str
    total_pages: int
    content_blocks: List[ContentBlock]
    signatures: List[SignatureComparison]
    financial_values: List[FinancialValue]
    redaction_count: int
    table_count: int
    overall_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DetailedSimilarityScore:
    """Detailed breakdown of similarity metrics between two signatures."""
    ssim_score: float = 0.0
    orb_score: float = 0.0
    hash_distance: int = 0
    hash_similarity: float = 0.0
    weighted_score: float = 0.0
    verdict: str = "Unknown"  # "Match", "Possible Match", "No Match"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ssim": round(self.ssim_score, 3),
            "orb": round(self.orb_score, 3),
            "hash_distance": self.hash_distance,
            "hash_similarity": round(self.hash_similarity, 3),
            "weighted_score": round(self.weighted_score, 3),
            "verdict": self.verdict
        }


class SignatureComparator:
    """
    Compares signatures to detect consistency or discrepancies.
    
    Uses a multi-metric approach:
    - SSIM (Structural Similarity Index) - 40% weight
    - ORB feature matching - 35% weight
    - Perceptual hash - 25% weight
    """
    
    # Similarity thresholds
    MATCH_THRESHOLD = 0.7
    POSSIBLE_MATCH_THRESHOLD = 0.5
    
    # Weighting for combined score
    SSIM_WEIGHT = 0.40
    ORB_WEIGHT = 0.35
    HASH_WEIGHT = 0.25
    
    def __init__(
        self,
        match_threshold: float = 0.7,
        possible_match_threshold: float = 0.5
    ):
        """
        Initialize the comparator.
        
        Args:
            match_threshold: Threshold for "Match" verdict (default 0.7)
            possible_match_threshold: Threshold for "Possible Match" (default 0.5)
        """
        self.signatures_by_name: Dict[str, List[SignatureBlock]] = defaultdict(list)
        self.MATCH_THRESHOLD = match_threshold
        self.POSSIBLE_MATCH_THRESHOLD = possible_match_threshold
        
        # Store detailed comparison results for the signature report
        self.detailed_comparisons: List[Dict[str, Any]] = []
    
    def add_signature(self, signature: SignatureBlock):
        """
        Add a signature to the comparator.
        
        Args:
            signature: SignatureBlock to add
        """
        # Group by name if available, otherwise by designation
        if signature.name:
            key = self._normalize_name(signature.name)
        elif signature.designation:
            key = f"[{signature.designation}]"
        else:
            key = "Unknown Signer"
        
        self.signatures_by_name[key].append(signature)
    
    def compare_all(self) -> List[SignatureComparison]:
        """
        Compare all signatures and return results.
        
        Returns:
            List of SignatureComparison results
        """
        results = []
        self.detailed_comparisons = []  # Reset detailed comparisons
        
        for name, signatures in self.signatures_by_name.items():
            comparison = self._compare_signatures(name, signatures)
            results.append(comparison)
        
        return results
    
    def get_detailed_comparisons(self) -> List[Dict[str, Any]]:
        """Get detailed pairwise comparison results."""
        return self.detailed_comparisons
    
    def _compare_signatures(
        self,
        name: str,
        signatures: List[SignatureBlock]
    ) -> SignatureComparison:
        """
        Compare signatures for a single person.
        
        Args:
            name: Person's name
            signatures: List of their signatures
            
        Returns:
            SignatureComparison result
        """
        pages = [sig.page_number for sig in signatures]
        designation = next(
            (sig.designation for sig in signatures if sig.designation),
            None
        )
        
        if len(signatures) == 1:
            return SignatureComparison(
                name=name,
                designation=designation,
                pages=pages,
                comparison_result="Single occurrence"
            )
        
        # Compare all pairs of signatures
        similarity_scores = []
        detailed_scores: List[DetailedSimilarityScore] = []
        all_match = True
        has_discrepancy = False
        
        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                sig1 = signatures[i]
                sig2 = signatures[j]
                
                # Compute detailed similarity
                if sig1.image is not None and sig2.image is not None:
                    detailed = self._compute_detailed_similarity(sig1.image, sig2.image)
                else:
                    # Fallback to description comparison
                    desc_sim = self._compare_descriptions(
                        sig1.visual_description,
                        sig2.visual_description
                    )
                    detailed = DetailedSimilarityScore(
                        weighted_score=desc_sim,
                        verdict=self._get_verdict(desc_sim)
                    )
                
                similarity_scores.append(detailed.weighted_score)
                detailed_scores.append(detailed)
                
                # Store detailed comparison
                self.detailed_comparisons.append({
                    "signer": name,
                    "signature_a_page": sig1.page_number,
                    "signature_b_page": sig2.page_number,
                    "scores": detailed.to_dict()
                })
                
                if detailed.verdict == "No Match":
                    has_discrepancy = True
                if detailed.verdict != "Match":
                    all_match = False
        
        # Determine overall result
        if has_discrepancy:
            comparison_result = "Discrepancy"
        elif all_match:
            comparison_result = "Consistent"
        else:
            comparison_result = "Possibly Consistent"
        
        # Generate notes
        notes = ""
        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            notes = f"Average similarity: {avg_score:.2f}"
            if has_discrepancy:
                notes += f" - Scores: {[f'{s:.2f}' for s in similarity_scores]}"
        
        return SignatureComparison(
            name=name,
            designation=designation,
            pages=pages,
            comparison_result=comparison_result,
            similarity_scores=similarity_scores,
            notes=notes
        )
    
    def _compute_detailed_similarity(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> DetailedSimilarityScore:
        """
        Compute detailed similarity between two signature images.
        
        Uses three metrics:
        1. SSIM (Structural Similarity Index)
        2. ORB feature matching
        3. Perceptual hash
        
        Args:
            img1: First signature image
            img2: Second signature image
            
        Returns:
            DetailedSimilarityScore with all metrics
        """
        import cv2
        
        # Resize images to same dimensions for fair comparison
        target_size = (200, 100)
        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)
        
        # Convert to grayscale
        if len(img1_resized.shape) == 3:
            gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1_resized
            
        if len(img2_resized.shape) == 3:
            gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2_resized
        
        # 1. Compute SSIM
        ssim_score = self._compute_ssim(gray1, gray2)
        
        # 2. Compute ORB feature similarity
        orb_score = self._compute_feature_similarity(gray1, gray2)
        
        # 3. Compute perceptual hash distance
        hash_distance = self._compute_hash_distance(gray1, gray2)
        # Convert hash distance to similarity (0-1 scale, max distance ~64 for 64-bit hash)
        hash_similarity = max(0, 1 - (hash_distance / 32))
        
        # Compute weighted score
        weighted_score = (
            self.SSIM_WEIGHT * ssim_score +
            self.ORB_WEIGHT * orb_score +
            self.HASH_WEIGHT * hash_similarity
        )
        
        # Determine verdict
        verdict = self._get_verdict(weighted_score)
        
        return DetailedSimilarityScore(
            ssim_score=ssim_score,
            orb_score=orb_score,
            hash_distance=hash_distance,
            hash_similarity=hash_similarity,
            weighted_score=weighted_score,
            verdict=verdict
        )
    
    def _get_verdict(self, score: float) -> str:
        """Get verdict based on similarity score."""
        if score >= self.MATCH_THRESHOLD:
            return "Match"
        elif score >= self.POSSIBLE_MATCH_THRESHOLD:
            return "Possible Match"
        else:
            return "No Match"
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM between two grayscale images.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            SSIM score (0-1)
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(img1, img2)
            return max(0, min(1, float(score)))
        except Exception:
            return 0.5
    
    def _compute_feature_similarity(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> float:
        """
        Compute feature-based similarity using ORB.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            Similarity score (0-1)
        """
        import cv2
        
        try:
            # Create ORB detector with more features for better matching
            orb = cv2.ORB_create(nfeatures=500)
            
            # Detect keypoints and compute descriptors
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return 0.5
            
            # Match descriptors using BFMatcher with cross-check
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if not matches:
                return 0.0
            
            # Sort by distance and use good matches
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 50]
            
            max_possible = min(len(kp1), len(kp2))
            if max_possible == 0:
                return 0.5
            
            similarity = len(good_matches) / max_possible
            return min(1.0, similarity)
            
        except Exception:
            return 0.5
    
    def _compute_hash_distance(self, img1: np.ndarray, img2: np.ndarray) -> int:
        """
        Compute perceptual hash distance between two images.
        
        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)
            
        Returns:
            Hamming distance between hashes (0 = identical)
        """
        hash1 = self._compute_phash(img1)
        hash2 = self._compute_phash(img2)
        
        # Compute Hamming distance
        return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
    
    def _compute_phash(self, image: np.ndarray, hash_size: int = 8) -> str:
        """
        Compute perceptual hash (pHash) of an image.
        
        Uses difference hash algorithm:
        1. Resize to (hash_size + 1, hash_size)
        2. Compute horizontal gradients
        3. Convert to binary hash
        
        Args:
            image: Grayscale image
            hash_size: Size of the hash (8x8 = 64 bits)
            
        Returns:
            Hexadecimal hash string
        """
        import cv2
        
        # Resize to hash_size + 1 for gradient computation
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        
        # Compute horizontal gradient (is pixel brighter than the next?)
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert to hex string
        hash_value = 0
        for row in diff:
            for val in row:
                hash_value = (hash_value << 1) | int(val)
        
        return format(hash_value, f'0{hash_size * hash_size // 4}x')
    
    def _compare_descriptions(self, desc1: str, desc2: str) -> float:
        """
        Compare signature visual descriptions using text similarity.
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Similarity score (0-1)
        """
        if not desc1 or not desc2:
            return 0.5  # Unknown
        
        # Normalize descriptions
        desc1_lower = desc1.lower().strip()
        desc2_lower = desc2.lower().strip()
        
        if desc1_lower == desc2_lower:
            return 1.0
        
        # Check for common keywords
        keywords1 = set(desc1_lower.split())
        keywords2 = set(desc2_lower.split())
        
        if not keywords1 or not keywords2:
            return 0.5
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.5
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Convert to title case, remove extra spaces
        normalized = ' '.join(name.split()).strip()
        # Remove common titles
        titles = ['mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'mr', 'mrs', 'ms', 'dr', 'prof']
        normalized_lower = normalized.lower()
        for title in titles:
            if normalized_lower.startswith(title + ' '):
                normalized = normalized[len(title):].strip()
                break
        return normalized.title() if normalized else "Unknown"


class FinancialValidator:
    """Validates and extracts financial values from text."""
    
    # Patterns for financial values
    PATTERNS = [
        # Currency + amount (e.g., "INR 500,000,000.00", "$1,234.56")
        (r'(?:INR|USD|EUR|GBP|JPY|AUD|CAD|SGD|\$|£|€|¥|Rs\.?)\s*[\d,]+(?:\.\d+)?', 'currency'),
        # Amount + currency (e.g., "500,000.00 INR")
        (r'[\d,]+(?:\.\d+)?\s*(?:INR|USD|EUR|GBP|JPY|AUD|CAD|SGD)', 'currency'),
        # Amount + unit (e.g., "500 Million", "50 Cr")
        (r'[\d,]+(?:\.\d+)?\s*(?:Million|Billion|Trillion|Crore|Lakh|Lakhs|Cr|Mn|Bn|Lac|Lacs)', 'amount_unit'),
        # Percentages (e.g., "15.5%", "8.75 %")
        (r'[\d,]+(?:\.\d+)?\s*%', 'percentage'),
        # Interest rates (e.g., "SOFR + 1.5%", "12.5% p.a.")
        (r'[\d,]+(?:\.\d+)?\s*%\s*(?:p\.?a\.?|per\s+annum)?', 'interest_rate'),
    ]
    
    def extract_financial_values(
        self,
        text: str,
        page: int,
        context_window: int = 50
    ) -> List[FinancialValue]:
        """
        Extract all financial values from text.
        
        Args:
            text: Text to search
            page: Page number
            context_window: Characters to include for context
            
        Returns:
            List of FinancialValue objects
        """
        values = []
        seen_values = set()  # Avoid duplicates
        
        for pattern, value_type in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group()
                
                # Skip if we've seen this exact value
                if value in seen_values:
                    continue
                seen_values.add(value)
                
                # Extract context
                start = max(0, match.start() - context_window)
                end = min(len(text), match.end() + context_window)
                context = text[start:end].strip()
                
                values.append(FinancialValue(
                    value=value,
                    page=page,
                    context=context,
                    value_type=value_type,
                    confidence=1.0
                ))
        
        return values
    
    def validate_consistency(
        self,
        values: List[FinancialValue]
    ) -> List[FinancialValue]:
        """
        Check for consistency of financial values across the document.
        
        If the same amount appears multiple times, verify they match.
        
        Args:
            values: List of extracted values
            
        Returns:
            List with confidence scores adjusted
        """
        # Group by normalized value
        value_occurrences: Dict[str, List[FinancialValue]] = defaultdict(list)
        
        for val in values:
            normalized = self._normalize_value(val.value)
            value_occurrences[normalized].append(val)
        
        # Adjust confidence for repeated values
        validated = []
        for normalized, occurrences in value_occurrences.items():
            if len(occurrences) > 1:
                # Multiple occurrences - check consistency
                values_match = all(
                    self._normalize_value(v.value) == normalized
                    for v in occurrences
                )
                confidence = 1.0 if values_match else 0.7
                
                for val in occurrences:
                    val.confidence = confidence
                    validated.append(val)
            else:
                validated.append(occurrences[0])
        
        return validated
    
    def _normalize_value(self, value: str) -> str:
        """Normalize a financial value for comparison."""
        # Remove extra spaces
        normalized = ' '.join(value.split())
        # Standardize currency symbols
        replacements = {
            'Rs.': 'INR',
            'Rs': 'INR',
            '$': 'USD',
            '£': 'GBP',
            '€': 'EUR',
            '¥': 'JPY',
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized


class Postprocessor:
    """
    Stage 3: Postprocessor
    
    Handles:
    1. Content merging in reading order
    2. Signature comparison across pages
    3. Financial value extraction and validation
    """
    
    def __init__(self):
        """Initialize postprocessor."""
        self.signature_comparator = SignatureComparator()
        self.financial_validator = FinancialValidator()
    
    def process(self, extraction: ExtractionResult) -> ProcessedDocument:
        """
        Post-process extraction results.
        
        Args:
            extraction: ExtractionResult from Stage 2
            
        Returns:
            ProcessedDocument ready for output
        """
        # 1. Merge content blocks in reading order
        content_blocks = self._merge_content(extraction)
        
        # 2. Process signatures
        signatures = self._process_signatures(extraction)
        
        # 3. Extract and validate financial values
        financial_values = self._process_financial_values(content_blocks)
        
        # 4. Count redactions and tables
        redaction_count = sum(
            len(page.redactions)
            for page in extraction.pages
        )
        table_count = sum(
            len(page.tables)
            for page in extraction.pages
        )
        
        return ProcessedDocument(
            source_file=extraction.source_file,
            provider=extraction.provider,
            total_pages=extraction.total_pages,
            content_blocks=content_blocks,
            signatures=signatures,
            financial_values=financial_values,
            redaction_count=redaction_count,
            table_count=table_count,
            overall_confidence=extraction.overall_confidence
        )
    
    def _merge_content(self, extraction: ExtractionResult) -> List[ContentBlock]:
        """
        Merge content from all pages in reading order.
        
        Args:
            extraction: ExtractionResult
            
        Returns:
            List of ContentBlock in document order
        """
        all_blocks = []
        
        for page in extraction.pages:
            page_blocks = self._extract_page_blocks(page)
            all_blocks.extend(page_blocks)
        
        # Sort by page, then by position (top to bottom, left to right)
        all_blocks.sort(key=lambda b: (b.page, b.position[0], b.position[1]))
        
        return all_blocks
    
    def _extract_page_blocks(self, page: PageExtraction) -> List[ContentBlock]:
        """
        Extract content blocks from a page.
        
        Args:
            page: PageExtraction
            
        Returns:
            List of ContentBlock
        """
        blocks = []
        
        # Add text and tables based on layout region positions
        layout_regions = sorted(
            page.layout.regions,
            key=lambda r: (r.bbox.y if r.bbox else 1e9, r.bbox.x if r.bbox else 1e9)
        )
        for region in layout_regions:
            if region.region_type == RegionType.TEXT_BLOCK and region.content and region.content.strip():
                position = (region.bbox.y, region.bbox.x) if region.bbox else (1e9, 1e9)
                blocks.append(ContentBlock(
                    block_type="text",
                    content=region.content,
                    page=page.page_number,
                    position=position
                ))
            elif region.region_type == RegionType.TABLE and region.table:
                table = region.table
                position = (table.bbox.y, table.bbox.x) if table.bbox else (1e9, 1e9)
                blocks.append(ContentBlock(
                    block_type="table",
                    content=table.markdown or table.to_markdown(),
                    page=page.page_number,
                    position=position,
                    metadata={"row_count": table.row_count, "column_count": table.column_count}
                ))
        
        # Add redactions
        for redaction in page.redactions:
            position = (redaction.y, redaction.x)
            
            blocks.append(ContentBlock(
                block_type="redacted",
                content="[REDACTED]",
                page=page.page_number,
                position=position
            ))
        
        # Add signatures (as markers in content)
        for sig in page.signatures:
            position = (0.0, 0.0)
            if sig.bbox:
                position = (sig.bbox.y, sig.bbox.x)
            
            sig_text = f"[Signature: {sig.name or 'Unknown'}]"
            if sig.designation:
                sig_text += f" - {sig.designation}"
            
            blocks.append(ContentBlock(
                block_type="signature",
                content=sig_text,
                page=page.page_number,
                position=position,
                metadata={
                    "name": sig.name,
                    "designation": sig.designation,
                    "date": sig.date,
                    "image_path": sig.image_path
                }
            ))
        
        return blocks
    
    def _process_signatures(self, extraction: ExtractionResult) -> List[SignatureComparison]:
        """
        Process and compare all signatures.
        
        Args:
            extraction: ExtractionResult
            
        Returns:
            List of SignatureComparison
        """
        # Add all signatures to comparator
        for page in extraction.pages:
            for sig in page.signatures:
                self.signature_comparator.add_signature(sig)
        
        # Compare signatures
        return self.signature_comparator.compare_all()
    
    def _process_financial_values(
        self,
        content_blocks: List[ContentBlock]
    ) -> List[FinancialValue]:
        """
        Extract and validate financial values from content.
        
        Args:
            content_blocks: List of content blocks
            
        Returns:
            List of validated FinancialValue
        """
        all_values = []
        
        for block in content_blocks:
            if block.block_type in ("text", "table"):
                values = self.financial_validator.extract_financial_values(
                    block.content,
                    block.page
                )
                all_values.extend(values)
        
        # Validate consistency
        validated = self.financial_validator.validate_consistency(all_values)
        
        return validated
