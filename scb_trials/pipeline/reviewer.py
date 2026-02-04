"""
Stage 4: Reviewer
Handles confidence scoring and output generation (Markdown/JSON).
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from .postprocessor import (
    ProcessedDocument,
    ContentBlock,
    SignatureComparison,
    FinancialValue,
    SignatureComparator,
)
from providers.base import ExtractionResult


@dataclass
class LowConfidenceRegion:
    """A region with low confidence that may need review."""
    page: int
    region_type: str
    confidence: float
    description: str


@dataclass
class ConfidenceReport:
    """Complete confidence report for a document."""
    overall_score: float
    page_scores: List[float]
    low_confidence_regions: List[LowConfidenceRegion]
    signature_analysis: Dict[str, str]
    financial_values_found: int
    tables_extracted: int
    redactions_found: int
    warnings: List[str] = field(default_factory=list)
    
    def needs_review(self, threshold: float = 0.8) -> bool:
        """Check if document needs manual review."""
        return (
            self.overall_score < threshold or
            len(self.low_confidence_regions) > 0 or
            any(v == "Discrepancy" for v in self.signature_analysis.values())
        )


class Reviewer:
    """
    Stage 4: Reviewer
    
    Handles:
    1. Confidence scoring across all extractions
    2. Generating final Markdown output
    3. Generating structured JSON output
    4. Identifying areas needing manual review
    """
    
    # Confidence threshold for flagging
    CONFIDENCE_THRESHOLD = 0.8
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize reviewer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
    
    def review(self, document: ProcessedDocument) -> ConfidenceReport:
        """
        Review the processed document and generate confidence report.
        
        Args:
            document: ProcessedDocument from postprocessor
            
        Returns:
            ConfidenceReport
        """
        # Calculate page-level confidence (placeholder - actual confidence comes from extraction)
        page_scores = self._calculate_page_scores(document)
        
        # Find low confidence regions
        low_confidence = self._find_low_confidence_regions(document)
        
        # Summarize signature analysis
        signature_analysis = {
            sig.name: sig.comparison_result
            for sig in document.signatures
        }
        
        # Generate warnings
        warnings = self._generate_warnings(document)
        
        return ConfidenceReport(
            overall_score=document.overall_confidence,
            page_scores=page_scores,
            low_confidence_regions=low_confidence,
            signature_analysis=signature_analysis,
            financial_values_found=len(document.financial_values),
            tables_extracted=document.table_count,
            redactions_found=document.redaction_count,
            warnings=warnings
        )
    
    def generate_markdown(
        self,
        document: ProcessedDocument,
        confidence_report: ConfidenceReport
    ) -> str:
        """
        Generate final Markdown output.
        
        Args:
            document: ProcessedDocument
            confidence_report: ConfidenceReport
            
        Returns:
            Markdown string
        """
        lines = []
        
        # Header
        lines.append(f"# Document Extraction Report")
        lines.append(f"")
        lines.append(f"**Source:** {document.source_file}")
        lines.append(f"**Provider:** {document.provider}")
        lines.append(f"**Pages:** {document.total_pages}")
        lines.append(f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Overall Confidence:** {document.overall_confidence:.1%}")
        lines.append(f"")
        
        # Signature Analysis Section (as required by spec)
        lines.append("---")
        lines.append("")
        lines.append("## Signature Analysis")
        lines.append("")
        
        if document.signatures:
            lines.append("| Name | Designation | Pages | Visual Comparison |")
            lines.append("|------|-------------|-------|-------------------|")
            
            for sig in document.signatures:
                name = sig.name or "Unknown"
                designation = sig.designation or "-"
                pages = ", ".join(str(p) for p in sig.pages)
                comparison = sig.comparison_result
                
                lines.append(f"| {name} | {designation} | {pages} | {comparison} |")
            
            lines.append("")
        else:
            lines.append("*No signatures detected in this document.*")
            lines.append("")
        
        # Financial Values Summary
        if document.financial_values:
            lines.append("### Financial Values Found")
            lines.append("")
            lines.append("| Value | Page | Type |")
            lines.append("|-------|------|------|")
            
            for fv in document.financial_values[:20]:  # Limit to first 20
                lines.append(f"| {fv.value} | {fv.page} | {fv.value_type} |")
            
            if len(document.financial_values) > 20:
                lines.append(f"| ... | ... | *{len(document.financial_values) - 20} more values* |")
            
            lines.append("")
        
        # Warnings
        if confidence_report.warnings:
            lines.append("### Warnings")
            lines.append("")
            for warning in confidence_report.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        # Document Content
        lines.append("---")
        lines.append("")
        lines.append("## Document Content")
        lines.append("")
        
        # Group content by page
        current_page = 0
        for block in document.content_blocks:
            if block.page != current_page:
                current_page = block.page
                lines.append(f"### Page {current_page}")
                lines.append("")
            
            if block.block_type == "text":
                lines.append(block.content)
                lines.append("")
            elif block.block_type == "table":
                lines.append(block.content)
                lines.append("")
            elif block.block_type == "redacted":
                lines.append("[REDACTED]")
                lines.append("")
            elif block.block_type == "signature":
                lines.append(f"*{block.content}*")
                lines.append("")
        
        # Confidence Report
        if confidence_report.needs_review():
            lines.append("---")
            lines.append("")
            lines.append("## Review Required")
            lines.append("")
            lines.append("This document has been flagged for manual review due to:")
            lines.append("")
            
            if confidence_report.overall_score < self.CONFIDENCE_THRESHOLD:
                lines.append(f"- Overall confidence ({confidence_report.overall_score:.1%}) below threshold")
            
            for region in confidence_report.low_confidence_regions:
                lines.append(f"- Low confidence {region.region_type} on page {region.page}: {region.description}")
            
            for name, result in confidence_report.signature_analysis.items():
                if result == "Discrepancy":
                    lines.append(f"- Signature discrepancy detected for: {name}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_json(
        self,
        document: ProcessedDocument,
        confidence_report: ConfidenceReport
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            document: ProcessedDocument
            confidence_report: ConfidenceReport
            
        Returns:
            Dictionary ready for JSON serialization
        """
        output = {
            "metadata": {
                "source_file": document.source_file,
                "provider": document.provider,
                "pages": document.total_pages,
                "extraction_timestamp": datetime.now().isoformat(),
            },
            "confidence": {
                "overall": round(document.overall_confidence, 4),
                "pages": [round(s, 4) for s in confidence_report.page_scores],
                "needs_review": confidence_report.needs_review()
            },
            "signatures": [
                {
                    "name": sig.name,
                    "designation": sig.designation,
                    "pages": sig.pages,
                    "comparison": sig.comparison_result,
                    "notes": sig.notes
                }
                for sig in document.signatures
            ],
            "financial_values": [
                {
                    "value": fv.value,
                    "page": fv.page,
                    "type": fv.value_type,
                    "context": fv.context[:100] if fv.context else "",
                    "confidence": round(fv.confidence, 4)
                }
                for fv in document.financial_values
            ],
            "redactions": {
                "count": document.redaction_count,
                "pages": list(set(
                    block.page for block in document.content_blocks
                    if block.block_type == "redacted"
                ))
            },
            "tables": {
                "count": document.table_count
            },
            "content": {
                "pages": self._structure_content_by_page(document)
            },
            "warnings": confidence_report.warnings
        }
        
        return output
    
    def save_outputs(
        self,
        document: ProcessedDocument,
        confidence_report: ConfidenceReport,
        base_name: Optional[str] = None,
        extraction: Optional[ExtractionResult] = None,
        comparator: Optional[SignatureComparator] = None,
        generate_signature_report: bool = True
    ) -> Dict[str, str]:
        """
        Save all outputs to files.
        
        Args:
            document: ProcessedDocument
            confidence_report: ConfidenceReport
            base_name: Base name for output files (default: source file name)
            extraction: Optional ExtractionResult for signature report generation
            comparator: Optional SignatureComparator with detailed comparisons
            generate_signature_report: Whether to generate signature analysis report
            
        Returns:
            Dictionary of output file paths
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine base name
        if base_name is None:
            base_name = Path(document.source_file).stem
        
        output_paths = {}
        
        # Save Markdown
        markdown_content = self.generate_markdown(document, confidence_report)
        markdown_path = self.output_dir / f"{base_name}_extracted.md"
        markdown_path.write_text(markdown_content, encoding='utf-8')
        output_paths["markdown"] = str(markdown_path)
        
        # Save JSON
        json_content = self.generate_json(document, confidence_report)
        json_path = self.output_dir / f"{base_name}_extracted.json"
        json_path.write_text(
            json.dumps(json_content, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        output_paths["json"] = str(json_path)
        
        # Save confidence report
        report_path = self.output_dir / f"{base_name}_confidence.json"
        report_data = {
            "overall_score": confidence_report.overall_score,
            "page_scores": confidence_report.page_scores,
            "low_confidence_regions": [
                {
                    "page": r.page,
                    "region_type": r.region_type,
                    "confidence": r.confidence,
                    "description": r.description
                }
                for r in confidence_report.low_confidence_regions
            ],
            "signature_analysis": confidence_report.signature_analysis,
            "statistics": {
                "financial_values_found": confidence_report.financial_values_found,
                "tables_extracted": confidence_report.tables_extracted,
                "redactions_found": confidence_report.redactions_found
            },
            "needs_review": confidence_report.needs_review(),
            "warnings": confidence_report.warnings
        }
        
        # Task 9.6: Add timing instrumentation to confidence report
        if extraction and extraction.metadata:
            timing_data = {}
            if 'timing_summary' in extraction.metadata:
                timing_data['summary'] = extraction.metadata['timing_summary']
            if 'timing' in extraction.metadata:
                timing_data['per_page'] = extraction.metadata['timing']
            if 'extraction_time_seconds' in extraction.metadata:
                timing_data['total_extraction_seconds'] = extraction.metadata['extraction_time_seconds']
            if 'parallel_processing' in extraction.metadata:
                timing_data['parallel_processing'] = extraction.metadata['parallel_processing']
            if timing_data:
                report_data['timing'] = timing_data
        
        report_path.write_text(
            json.dumps(report_data, indent=2),
            encoding='utf-8'
        )
        output_paths["confidence_report"] = str(report_path)
        
        # Generate signature analysis report if requested and extraction is provided
        if generate_signature_report and extraction:
            sig_report_paths = self._generate_signature_report(
                extraction,
                comparator
            )
            output_paths.update(sig_report_paths)
        
        return output_paths
    
    def _generate_signature_report(
        self,
        extraction: ExtractionResult,
        comparator: Optional[SignatureComparator] = None
    ) -> Dict[str, str]:
        """
        Generate signature analysis report.
        
        Args:
            extraction: ExtractionResult from the pipeline
            comparator: Optional SignatureComparator with detailed comparisons
            
        Returns:
            Dictionary with paths to signature report files
        """
        from .signature_report import (
            SignatureReportGenerator,
            SignatureReportConfig,
        )
        
        config = SignatureReportConfig(
            include_base64_images=True,
            include_comparison_matrix=True,
            generate_markdown=True,
            generate_json=True
        )
        
        generator = SignatureReportGenerator(config)
        
        detailed_comparisons = None
        if comparator:
            detailed_comparisons = comparator.get_detailed_comparisons()
        
        report_paths = generator.generate_report(
            extraction,
            str(self.output_dir),
            detailed_comparisons
        )
        
        # Rename keys to avoid confusion with main outputs
        return {
            "signature_report_json": report_paths.get("json", ""),
            "signature_report_md": report_paths.get("md", "")
        }
    
    def _calculate_page_scores(self, document: ProcessedDocument) -> List[float]:
        """Calculate confidence scores for each page."""
        # Group content blocks by page
        page_blocks: Dict[int, List[ContentBlock]] = {}
        for block in document.content_blocks:
            if block.page not in page_blocks:
                page_blocks[block.page] = []
            page_blocks[block.page].append(block)
        
        # Calculate score for each page
        scores = []
        for page_num in range(1, document.total_pages + 1):
            blocks = page_blocks.get(page_num, [])
            
            if not blocks:
                # No content on page - might be blank or failed extraction
                scores.append(0.5)
            else:
                # Use overall confidence as base
                scores.append(document.overall_confidence)
        
        return scores
    
    def _find_low_confidence_regions(
        self,
        document: ProcessedDocument
    ) -> List[LowConfidenceRegion]:
        """Find regions with confidence below threshold."""
        low_confidence = []
        
        # Check financial values
        for fv in document.financial_values:
            if fv.confidence < self.CONFIDENCE_THRESHOLD:
                low_confidence.append(LowConfidenceRegion(
                    page=fv.page,
                    region_type="financial_value",
                    confidence=fv.confidence,
                    description=f"Financial value '{fv.value}' may need verification"
                ))
        
        # Check for signature discrepancies
        for sig in document.signatures:
            if sig.comparison_result == "Discrepancy":
                for page in sig.pages:
                    low_confidence.append(LowConfidenceRegion(
                        page=page,
                        region_type="signature",
                        confidence=0.5,
                        description=f"Signature for '{sig.name}' shows discrepancy"
                    ))
        
        return low_confidence
    
    def _generate_warnings(self, document: ProcessedDocument) -> List[str]:
        """Generate warnings for potential issues."""
        warnings = []
        
        # Check for signature discrepancies
        for sig in document.signatures:
            if sig.comparison_result == "Discrepancy":
                warnings.append(
                    f"Signature discrepancy detected for '{sig.name}' across pages {sig.pages}"
                )
        
        # Check for many redactions
        if document.redaction_count > 5:
            warnings.append(
                f"Document contains {document.redaction_count} redacted sections"
            )
        
        # Check for low overall confidence
        if document.overall_confidence < self.CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Overall extraction confidence ({document.overall_confidence:.1%}) is below recommended threshold"
            )
        
        # Check for empty pages
        page_content = {}
        for block in document.content_blocks:
            if block.page not in page_content:
                page_content[block.page] = []
            page_content[block.page].append(block)
        
        for page_num in range(1, document.total_pages + 1):
            if page_num not in page_content:
                warnings.append(f"Page {page_num} appears to have no extractable content")
        
        return warnings
    
    def _structure_content_by_page(
        self,
        document: ProcessedDocument
    ) -> List[Dict[str, Any]]:
        """Structure content blocks by page for JSON output."""
        page_content: Dict[int, List[Dict[str, Any]]] = {}
        
        for block in document.content_blocks:
            if block.page not in page_content:
                page_content[block.page] = []
            
            block_data = {
                "type": block.block_type,
                "content": block.content
            }
            
            if block.metadata:
                block_data["metadata"] = block.metadata
            
            page_content[block.page].append(block_data)
        
        # Convert to list format
        pages = []
        for page_num in range(1, document.total_pages + 1):
            pages.append({
                "page_number": page_num,
                "blocks": page_content.get(page_num, [])
            })
        
        return pages
