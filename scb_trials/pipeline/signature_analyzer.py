"""
Signature Analyzer Module
Provides enhanced signature detection, analysis, and comparison functionality.
"""

import base64
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from providers.base import BoundingBox, SignatureBlock


@dataclass
class SignatureSnippet:
    """Enhanced signature data with image snippet."""
    signature_id: str           # Unique identifier (e.g., "sig_p01_001")
    page_number: int
    bbox: Optional[BoundingBox] = None
    image_path: str = ""        # Path to saved PNG
    image_base64: str = ""      # Base64 encoded for JSON output
    image: Optional[np.ndarray] = None  # Raw image data
    name: Optional[str] = None
    designation: Optional[str] = None
    date: Optional[str] = None
    visual_description: str = ""
    confidence: float = 0.0
    context_text: str = ""      # Surrounding text for context
    
    def to_dict(self, include_image: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.signature_id,
            "page": self.page_number,
            "bbox": {
                "x": self.bbox.x if self.bbox else 0,
                "y": self.bbox.y if self.bbox else 0,
                "width": self.bbox.width if self.bbox else 0,
                "height": self.bbox.height if self.bbox else 0
            } if self.bbox else None,
            "image_path": self.image_path,
            "name": self.name,
            "designation": self.designation,
            "date": self.date,
            "visual_description": self.visual_description,
            "confidence": round(self.confidence, 3),
            "context": self.context_text
        }
        if include_image and self.image_base64:
            result["image_base64"] = self.image_base64
        return result
    
    @classmethod
    def from_signature_block(
        cls,
        sig_block: SignatureBlock,
        signature_id: str,
        context_text: str = ""
    ) -> "SignatureSnippet":
        """Create SignatureSnippet from a SignatureBlock."""
        snippet = cls(
            signature_id=signature_id,
            page_number=sig_block.page_number,
            bbox=sig_block.bbox,
            image_path=sig_block.image_path or "",
            image=sig_block.image,
            name=sig_block.name,
            designation=sig_block.designation,
            date=sig_block.date,
            visual_description=sig_block.visual_description,
            confidence=sig_block.confidence,
            context_text=context_text
        )
        
        # Generate base64 if image is available
        if sig_block.image is not None:
            snippet.image_base64 = snippet._encode_image_base64(sig_block.image)
        
        return snippet
    
    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 string."""
        import cv2
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer.tobytes()).decode('utf-8')


@dataclass
class SignatureOutlier:
    """Represents a signature that differs from others in its group."""
    signature_id: str
    page: int
    avg_similarity: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signature_id": self.signature_id,
            "page": self.page,
            "avg_similarity": round(self.avg_similarity, 2),
            "reason": self.reason
        }


@dataclass
class SignatureGroup:
    """Signatures grouped by person/role."""
    identifier: str             # Name or designation used for grouping
    designation: Optional[str] = None
    signatures: List[SignatureSnippet] = field(default_factory=list)
    internal_consistency: str = "Unknown"  # "Consistent", "Inconsistent", "Unknown"
    average_similarity: float = 0.0
    outliers: List[SignatureOutlier] = field(default_factory=list)  # Signatures that differ from the group
    
    @property
    def signature_ids(self) -> List[str]:
        """Get list of signature IDs in this group."""
        return [sig.signature_id for sig in self.signatures]
    
    @property
    def pages(self) -> List[int]:
        """Get list of pages where signatures appear."""
        return sorted(set(sig.page_number for sig in self.signatures))
    
    def to_dict(self, include_images: bool = False, compact: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        if compact:
            return {
                "role": self.designation or self.identifier,
                "count": len(self.signatures),
                "pages": self.pages,
                "consistency_score": round(self.average_similarity, 2) if self.average_similarity > 0 else None,
                "discrepancies": [o.to_dict() for o in self.outliers]
            }
        return {
            "identifier": self.identifier,
            "designation": self.designation,
            "signature_ids": self.signature_ids,
            "pages": self.pages,
            "count": len(self.signatures),
            "internal_consistency": self.internal_consistency,
            "average_similarity": round(self.average_similarity, 3) if self.average_similarity > 0 else None,
            "outliers": [o.to_dict() for o in self.outliers]
        }


@dataclass
class SignaturePairComparison:
    """Pairwise comparison result between two signatures."""
    signature_a_id: str
    signature_b_id: str
    similarity_score: float     # 0.0 to 1.0 (weighted combination)
    ssim_score: float = 0.0
    orb_score: float = 0.0
    hash_distance: int = 0
    match_verdict: str = "Unknown"  # "Match", "Possible Match", "No Match"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signature_a": self.signature_a_id,
            "signature_b": self.signature_b_id,
            "similarity_score": round(self.similarity_score, 3),
            "breakdown": {
                "ssim": round(self.ssim_score, 3),
                "orb": round(self.orb_score, 3),
                "hash_distance": self.hash_distance
            },
            "verdict": self.match_verdict
        }


@dataclass
class SignatureAnalysisReport:
    """Complete signature analysis report."""
    source_file: str
    extraction_date: str
    total_signatures: int = 0
    unique_signers: int = 0
    signatures: List[SignatureSnippet] = field(default_factory=list)
    signature_groups: List[SignatureGroup] = field(default_factory=list)
    comparisons: List[SignaturePairComparison] = field(default_factory=list)
    comparison_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self, include_images: bool = True, compact: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Args:
            include_images: Include base64 images in output (ignored in compact mode)
            compact: If True, return simplified summary-only format
            
        Returns:
            Dictionary representation of the report
        """
        if compact:
            # Compact mode: simplified summary with just counts, pages, consistency, discrepancies
            return {
                "report_metadata": {
                    "source_file": self.source_file,
                    "extraction_date": self.extraction_date,
                    "total_signatures": self.total_signatures,
                    "unique_signers": self.unique_signers
                },
                "signature_summary": [grp.to_dict(compact=True) for grp in self.signature_groups]
            }
        
        # Full mode: include all details
        return {
            "report_metadata": {
                "source_file": self.source_file,
                "extraction_date": self.extraction_date,
                "total_signatures_detected": self.total_signatures,
                "unique_signers_identified": self.unique_signers
            },
            "signatures": [sig.to_dict(include_image=include_images) for sig in self.signatures],
            "signature_groups": [grp.to_dict() for grp in self.signature_groups],
            "comparisons": [cmp.to_dict() for cmp in self.comparisons],
            "comparison_matrix": self.comparison_matrix,
            "summary": self.summary
        }


class SignatureAnalyzer:
    """
    Analyzes signatures from extracted document data.
    
    Handles:
    - Grouping signatures by signer (name/designation)
    - Pairwise comparison of signatures
    - Generating comparison matrix
    - Creating analysis summary
    """
    
    # Similarity thresholds for verdict
    MATCH_THRESHOLD = 0.7
    POSSIBLE_MATCH_THRESHOLD = 0.5
    
    # Weighting for combined score
    SSIM_WEIGHT = 0.40
    ORB_WEIGHT = 0.35
    HASH_WEIGHT = 0.25
    
    def __init__(self):
        """Initialize the analyzer."""
        self.signatures: List[SignatureSnippet] = []
        self.groups: Dict[str, SignatureGroup] = {}
    
    def add_signature(self, snippet: SignatureSnippet):
        """Add a signature snippet to the analyzer."""
        self.signatures.append(snippet)
        
        # Group by identifier (name or designation)
        identifier = self._get_grouping_key(snippet)
        if identifier not in self.groups:
            self.groups[identifier] = SignatureGroup(
                identifier=identifier,
                designation=snippet.designation
            )
        self.groups[identifier].signatures.append(snippet)
    
    def _get_grouping_key(self, snippet: SignatureSnippet) -> str:
        """
        Get the key used for grouping signatures.
        
        Priority: name > designation > "Unknown"
        """
        if snippet.name and snippet.name.strip():
            return self._normalize_name(snippet.name)
        if snippet.designation and snippet.designation.strip():
            return f"[{snippet.designation}]"
        return "Unknown Signer"
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Remove extra whitespace, convert to title case
        normalized = ' '.join(name.split()).strip()
        return normalized.title() if normalized else "Unknown"
    
    def analyze(self, source_file: str) -> SignatureAnalysisReport:
        """
        Perform full signature analysis.
        
        Args:
            source_file: Name of the source document
            
        Returns:
            SignatureAnalysisReport with all analysis results
        """
        from datetime import datetime
        
        # Perform pairwise comparisons within each group
        all_comparisons = []
        comparison_matrix: Dict[str, Dict[str, float]] = {}
        
        for group in self.groups.values():
            if len(group.signatures) > 1:
                group_comparisons = self._compare_group_signatures(group)
                all_comparisons.extend(group_comparisons)
                
                # Update group consistency based on comparisons
                if group_comparisons:
                    avg_score = sum(c.similarity_score for c in group_comparisons) / len(group_comparisons)
                    group.average_similarity = avg_score
                    
                    if avg_score >= self.MATCH_THRESHOLD:
                        group.internal_consistency = "Consistent"
                    elif avg_score >= self.POSSIBLE_MATCH_THRESHOLD:
                        group.internal_consistency = "Possibly Consistent"
                    else:
                        group.internal_consistency = "Inconsistent"
                
                # Detect outliers using cluster-based analysis
                group.outliers = self._detect_outliers(group, group_comparisons)
                        
                # Build comparison matrix
                for comp in group_comparisons:
                    if comp.signature_a_id not in comparison_matrix:
                        comparison_matrix[comp.signature_a_id] = {}
                    comparison_matrix[comp.signature_a_id][comp.signature_b_id] = comp.similarity_score
            else:
                group.internal_consistency = "Single Occurrence"
        
        # Generate summary
        summary = self._generate_summary()
        
        return SignatureAnalysisReport(
            source_file=source_file,
            extraction_date=datetime.now().isoformat(),
            total_signatures=len(self.signatures),
            unique_signers=len(self.groups),
            signatures=self.signatures,
            signature_groups=list(self.groups.values()),
            comparisons=all_comparisons,
            comparison_matrix=comparison_matrix,
            summary=summary
        )
    
    def _compare_group_signatures(
        self,
        group: SignatureGroup
    ) -> List[SignaturePairComparison]:
        """
        Compare all pairs of signatures within a group.
        
        Args:
            group: SignatureGroup to analyze
            
        Returns:
            List of pairwise comparisons
        """
        comparisons = []
        sigs = group.signatures
        
        for i in range(len(sigs) - 1):
            for j in range(i + 1, len(sigs)):
                sig_a = sigs[i]
                sig_b = sigs[j]
                
                # Skip if either signature lacks image data
                if sig_a.image is None or sig_b.image is None:
                    continue
                
                comparison = self._compare_signatures(sig_a, sig_b)
                comparisons.append(comparison)
        
        return comparisons
    
    def _detect_outliers(
        self,
        group: SignatureGroup,
        comparisons: List[SignaturePairComparison]
    ) -> List[SignatureOutlier]:
        """
        Detect outlier signatures using cluster-based analysis.
        
        A signature is considered an outlier if its average similarity
        to other signatures in the group is significantly below the group average.
        
        Args:
            group: SignatureGroup to analyze
            comparisons: Pre-computed pairwise comparisons
            
        Returns:
            List of SignatureOutlier objects
        """
        if len(group.signatures) < 3:
            return []  # Need at least 3 signatures to detect outliers
        
        if not comparisons:
            return []  # No comparisons available (no images)
        
        # Build similarity matrix from existing comparisons
        similarities: Dict[str, List[float]] = defaultdict(list)
        
        for comp in comparisons:
            similarities[comp.signature_a_id].append(comp.similarity_score)
            similarities[comp.signature_b_id].append(comp.similarity_score)
        
        if not similarities:
            return []
        
        # Compute average similarity per signature
        avg_scores: Dict[str, float] = {}
        for sig_id, scores in similarities.items():
            if scores:
                avg_scores[sig_id] = np.mean(scores)
        
        if not avg_scores:
            return []
        
        # Compute group statistics
        all_avgs = list(avg_scores.values())
        group_avg = np.mean(all_avgs)
        std = np.std(all_avgs)
        
        # Threshold: below group_avg - 1.5 * std, but at least 0.4
        threshold = max(0.4, group_avg - 1.5 * std)
        
        # Find outliers
        outliers = []
        for sig in group.signatures:
            sig_id = sig.signature_id
            if sig_id in avg_scores and avg_scores[sig_id] < threshold:
                outliers.append(SignatureOutlier(
                    signature_id=sig_id,
                    page=sig.page_number,
                    avg_similarity=avg_scores[sig_id],
                    reason=f"Low similarity ({avg_scores[sig_id]:.2f}) vs group avg ({group_avg:.2f})"
                ))
        
        return outliers
    
    def _compare_signatures(
        self,
        sig_a: SignatureSnippet,
        sig_b: SignatureSnippet
    ) -> SignaturePairComparison:
        """
        Compare two signatures and return detailed comparison result.
        
        Args:
            sig_a: First signature
            sig_b: Second signature
            
        Returns:
            SignaturePairComparison with all metrics
        """
        import cv2
        
        img_a = sig_a.image
        img_b = sig_b.image
        
        # Compute individual metrics
        ssim_score = self._compute_ssim(img_a, img_b)
        orb_score = self._compute_orb_similarity(img_a, img_b)
        hash_distance = self._compute_hash_distance(img_a, img_b)
        
        # Convert hash distance to similarity (0-1 scale)
        # Max hash distance for 64-bit hash is 64
        hash_similarity = max(0, 1 - (hash_distance / 32))
        
        # Compute weighted score
        weighted_score = (
            self.SSIM_WEIGHT * ssim_score +
            self.ORB_WEIGHT * orb_score +
            self.HASH_WEIGHT * hash_similarity
        )
        
        # Determine verdict
        if weighted_score >= self.MATCH_THRESHOLD:
            verdict = "Match"
        elif weighted_score >= self.POSSIBLE_MATCH_THRESHOLD:
            verdict = "Possible Match"
        else:
            verdict = "No Match"
        
        return SignaturePairComparison(
            signature_a_id=sig_a.signature_id,
            signature_b_id=sig_b.signature_id,
            similarity_score=weighted_score,
            ssim_score=ssim_score,
            orb_score=orb_score,
            hash_distance=hash_distance,
            match_verdict=verdict
        )
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            SSIM score (0-1)
        """
        import cv2
        
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2
            
            # Resize to same dimensions
            h1, w1 = gray1.shape
            h2, w2 = gray2.shape
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            
            gray1_resized = cv2.resize(gray1, (target_w, target_h))
            gray2_resized = cv2.resize(gray2, (target_w, target_h))
            
            # Compute SSIM
            score = ssim(gray1_resized, gray2_resized)
            return max(0, min(1, score))
            
        except Exception:
            return 0.5  # Default score on error
    
    def _compute_orb_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity using ORB feature matching.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            Similarity score (0-1) based on matching features
        """
        import cv2
        
        try:
            # Convert to grayscale
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2
            
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=500)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return 0.5
            
            # Match descriptors using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Sort by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use ratio of good matches
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
            img1: First image
            img2: Second image
            
        Returns:
            Hamming distance between hashes (0 = identical)
        """
        hash1 = self._compute_phash(img1)
        hash2 = self._compute_phash(img2)
        
        # Compute Hamming distance
        return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
    
    def _compute_phash(self, image: np.ndarray, hash_size: int = 8) -> str:
        """
        Compute perceptual hash of an image.
        
        Args:
            image: Source image
            hash_size: Size of the hash (8x8 = 64 bits)
            
        Returns:
            Hexadecimal hash string
        """
        import cv2
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to hash_size + 1 for gradient computation
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Compute horizontal gradient
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert to hex string
        hash_value = 0
        for row in diff:
            for val in row:
                hash_value = (hash_value << 1) | int(val)
        
        return format(hash_value, f'0{hash_size * hash_size // 4}x')
    
    def _generate_summary(self) -> Dict[str, str]:
        """Generate human-readable summary for each signer."""
        summary = {}
        
        for identifier, group in self.groups.items():
            pages_str = ", ".join(str(p) for p in group.pages)
            count = len(group.signatures)
            
            if count == 1:
                summary[identifier] = f"1 signature found (page {pages_str}) - Single occurrence"
            else:
                consistency = group.internal_consistency
                avg_score = group.average_similarity
                
                if avg_score > 0:
                    summary[identifier] = (
                        f"{count} signatures found (pages {pages_str}) - "
                        f"{consistency} (avg: {avg_score:.2f})"
                    )
                else:
                    summary[identifier] = (
                        f"{count} signatures found (pages {pages_str}) - "
                        f"Unable to verify consistency"
                    )
        
        return summary
