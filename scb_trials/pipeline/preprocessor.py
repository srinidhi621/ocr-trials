"""
Stage 1: Preprocessor
Handles PDF to image conversion and image enhancement.
"""

import os
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing operations."""
    dpi: int = 300  # DPI for PDF to image conversion
    enhance: bool = True  # Whether to apply image enhancement
    deskew: bool = True  # Whether to correct skew
    denoise: bool = True  # Whether to apply denoising
    contrast_enhance: bool = True  # Whether to enhance contrast
    output_format: str = "png"  # Output image format
    save_intermediates: bool = False  # Whether to save intermediate images
    output_dir: Optional[str] = None  # Directory for intermediate outputs


@dataclass
class PageImage:
    """Represents a processed page image."""
    page_number: int
    image: np.ndarray  # OpenCV image (BGR format)
    width: int
    height: int
    original_image: Optional[np.ndarray] = None  # Before enhancement
    skew_angle: float = 0.0
    enhancement_applied: bool = False
    
    def to_bytes(self, format: str = "png") -> bytes:
        """Convert image to bytes."""
        import cv2
        
        if format.lower() == "png":
            _, buffer = cv2.imencode('.png', self.image)
        elif format.lower() in ("jpg", "jpeg"):
            _, buffer = cv2.imencode('.jpg', self.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            _, buffer = cv2.imencode('.png', self.image)
        
        return buffer.tobytes()
    
    def to_pil(self):
        """Convert to PIL Image."""
        import cv2
        from PIL import Image
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


class Preprocessor:
    """
    Stage 1: Preprocessor
    
    Handles:
    1. PDF to image conversion (high DPI for quality)
    2. Image enhancement (deskew, denoise, contrast)
    3. Preparation for layout analysis
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or PreprocessConfig()
    
    def process_pdf(self, pdf_path: str) -> List[PageImage]:
        """
        Convert PDF to images and apply preprocessing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PageImage objects, one per page
        """
        # Convert PDF to images
        raw_images = self._pdf_to_images(pdf_path)
        
        # Process each page
        processed_pages = []
        for page_num, raw_image in enumerate(raw_images, start=1):
            page = self._process_page(raw_image, page_num)
            processed_pages.append(page)
            
            # Save intermediate if configured
            if self.config.save_intermediates and self.config.output_dir:
                self._save_intermediate(page, page_num)
        
        return processed_pages
    
    def process_image(self, image_path: str) -> PageImage:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PageImage object
        """
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self._process_page(image, 1)
    
    def process_image_bytes(self, image_bytes: bytes, page_number: int = 1) -> PageImage:
        """
        Process image from bytes.
        
        Args:
            image_bytes: Image as bytes
            page_number: Page number for reference
            
        Returns:
            PageImage object
        """
        import cv2
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image bytes")
        
        return self._process_page(image, page_number)
    
    def _pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images using pdf2image.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images as numpy arrays (BGR format)
        """
        import cv2
        from pdf2image import convert_from_path
        
        # Convert PDF to PIL images
        pil_images = convert_from_path(
            pdf_path,
            dpi=self.config.dpi,
            fmt=self.config.output_format
        )
        
        # Convert to OpenCV format (BGR)
        cv_images = []
        for pil_image in pil_images:
            # Convert PIL to numpy array (RGB)
            np_image = np.array(pil_image)
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            cv_images.append(bgr_image)
        
        return cv_images
    
    def _process_page(self, image: np.ndarray, page_number: int) -> PageImage:
        """
        Apply preprocessing to a single page image.
        
        Args:
            image: Image as numpy array (BGR format)
            page_number: Page number for reference
            
        Returns:
            PageImage object with processed image
        """
        original = image.copy()
        processed = image.copy()
        skew_angle = 0.0
        enhancement_applied = False
        
        if self.config.enhance:
            # Deskew
            if self.config.deskew:
                processed, skew_angle = self._deskew(processed)
            
            # Denoise
            if self.config.denoise:
                processed = self._denoise(processed)
            
            # Contrast enhancement
            if self.config.contrast_enhance:
                processed = self._enhance_contrast(processed)
            
            enhancement_applied = True
        
        height, width = processed.shape[:2]
        
        return PageImage(
            page_number=page_number,
            image=processed,
            width=width,
            height=height,
            original_image=original if self.config.save_intermediates else None,
            skew_angle=skew_angle,
            enhancement_applied=enhancement_applied
        )
    
    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew in the image.
        
        Uses Hough line detection to find dominant angle.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (corrected image, skew angle in degrees)
        """
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return image, 0.0
        
        # Calculate dominant angle
        angles = []
        for line in lines[:20]:  # Use top 20 lines
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # Normalize to -45 to 45 degrees
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)
        
        # Use median angle to be robust to outliers
        if angles:
            skew_angle = np.median(angles)
        else:
            return image, 0.0
        
        # Only correct if skew is significant (> 0.5 degrees)
        if abs(skew_angle) < 0.5:
            return image, 0.0
        
        # Rotate image to correct skew
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        
        # Calculate new dimensions to prevent cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated, skew_angle
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to the image.
        
        Uses Non-local Means Denoising for colored images.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Denoised image
        """
        import cv2
        
        # Non-local means denoising
        # Parameters: src, h (filter strength), hForColorComponents, templateWindowSize, searchWindowSize
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,  # Luminance filter strength
            hColor=10,  # Color filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Contrast-enhanced image
        """
        import cv2
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _save_intermediate(self, page: PageImage, page_number: int):
        """Save intermediate processing results."""
        import cv2
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed image
        processed_path = output_dir / f"page_{page_number:03d}_processed.{self.config.output_format}"
        cv2.imwrite(str(processed_path), page.image)
        
        # Save original if available
        if page.original_image is not None:
            original_path = output_dir / f"page_{page_number:03d}_original.{self.config.output_format}"
            cv2.imwrite(str(original_path), page.original_image)
    
    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """
        Get the number of pages in a PDF without converting.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
