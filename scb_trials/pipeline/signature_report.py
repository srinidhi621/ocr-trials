"""
Signature Report Generator
Creates detailed JSON and Markdown reports for signature analysis.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from providers.base import ExtractionResult, SignatureBlock
from .signature_analyzer import (
    SignatureSnippet,
    SignatureGroup,
    SignaturePairComparison,
    SignatureAnalysisReport,
    SignatureAnalyzer,
)
from .postprocessor import SignatureComparator


@dataclass
class SignatureReportConfig:
    """Configuration for signature report generation."""
    include_base64_images: bool = True  # Include base64 encoded images in JSON
    include_comparison_matrix: bool = True  # Include full comparison matrix
    signatures_dir: str = "signatures"  # Directory name for signature images
    generate_markdown: bool = True  # Generate Markdown report
    generate_json: bool = True  # Generate JSON report
    compact_mode: bool = True  # Generate simplified summary-only reports


class SignatureReportGenerator:
    """
    Generates comprehensive signature analysis reports.
    
    Creates:
    1. JSON report with all signature data and comparisons
    2. Markdown report with human-readable analysis
    """
    
    def __init__(self, config: Optional[SignatureReportConfig] = None):
        """
        Initialize the report generator.
        
        Args:
            config: Report configuration options
        """
        self.config = config or SignatureReportConfig()
    
    def generate_report(
        self,
        extraction: ExtractionResult,
        output_dir: str,
        detailed_comparisons: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Generate signature analysis report from extraction results.
        
        Args:
            extraction: ExtractionResult from the pipeline
            output_dir: Directory to save reports
            detailed_comparisons: Optional pre-computed comparison results
            
        Returns:
            Dict with paths to generated files {"json": path, "md": path}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base filename from source
        source_name = Path(extraction.source_file).stem
        
        # Collect all signatures from extraction
        all_signatures: List[SignatureBlock] = []
        for page in extraction.pages:
            all_signatures.extend(page.signatures)
        
        # Create analyzer and add signatures
        analyzer = SignatureAnalyzer()
        sig_counter = 0
        
        for sig in all_signatures:
            # Generate unique signature ID
            sig_id = f"sig_p{sig.page_number:02d}_{sig_counter:03d}"
            sig_counter += 1
            
            # Convert to SignatureSnippet
            snippet = SignatureSnippet.from_signature_block(
                sig,
                signature_id=sig_id,
                context_text=self._extract_context(sig)
            )
            
            analyzer.add_signature(snippet)
        
        # Generate analysis report
        report = analyzer.analyze(extraction.source_file)
        
        # Add detailed comparisons if provided
        if detailed_comparisons:
            report = self._merge_detailed_comparisons(report, detailed_comparisons)
        
        generated_files = {}
        
        # Generate JSON report
        if self.config.generate_json:
            json_path = output_path / f"{source_name}_signature_report.json"
            self._write_json_report(report, json_path)
            generated_files["json"] = str(json_path)
        
        # Generate Markdown report
        if self.config.generate_markdown:
            md_path = output_path / f"{source_name}_signature_report.md"
            self._write_markdown_report(report, md_path, output_dir)
            generated_files["md"] = str(md_path)
        
        return generated_files
    
    def _extract_context(self, sig: SignatureBlock) -> str:
        """Extract context text for a signature (e.g., nearby text)."""
        context_parts = []
        if sig.name:
            context_parts.append(f"Name: {sig.name}")
        if sig.designation:
            context_parts.append(f"Title: {sig.designation}")
        if sig.date:
            context_parts.append(f"Date: {sig.date}")
        return " | ".join(context_parts) if context_parts else ""
    
    def _merge_detailed_comparisons(
        self,
        report: SignatureAnalysisReport,
        detailed_comparisons: List[Dict[str, Any]]
    ) -> SignatureAnalysisReport:
        """Merge detailed comparison results into the report."""
        # Convert detailed comparisons to SignaturePairComparison objects
        for comp_data in detailed_comparisons:
            scores = comp_data.get("scores", {})
            
            comparison = SignaturePairComparison(
                signature_a_id=f"page_{comp_data.get('signature_a_page', 0)}",
                signature_b_id=f"page_{comp_data.get('signature_b_page', 0)}",
                similarity_score=scores.get("weighted_score", 0),
                ssim_score=scores.get("ssim", 0),
                orb_score=scores.get("orb", 0),
                hash_distance=scores.get("hash_distance", 0),
                match_verdict=scores.get("verdict", "Unknown")
            )
            
            # Avoid duplicates
            if not any(
                c.signature_a_id == comparison.signature_a_id and 
                c.signature_b_id == comparison.signature_b_id
                for c in report.comparisons
            ):
                report.comparisons.append(comparison)
        
        return report
    
    def _write_json_report(
        self,
        report: SignatureAnalysisReport,
        output_path: Path
    ):
        """Write JSON report to file."""
        if self.config.compact_mode:
            # Compact mode: simplified summary-only format
            report_dict = report.to_dict(compact=True)
        else:
            # Full mode: include all details
            report_dict = report.to_dict(include_images=self.config.include_base64_images)
            
            # Clean up the report for JSON serialization
            if not self.config.include_comparison_matrix:
                report_dict.pop("comparison_matrix", None)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    def _write_markdown_report(
        self,
        report: SignatureAnalysisReport,
        output_path: Path,
        output_dir: str
    ):
        """Write Markdown report to file."""
        if self.config.compact_mode:
            self._write_compact_markdown_report(report, output_path, output_dir)
        else:
            self._write_full_markdown_report(report, output_path, output_dir)
    
    def _write_compact_markdown_report(
        self,
        report: SignatureAnalysisReport,
        output_path: Path,
        output_dir: str
    ):
        """Write concise Markdown report with discrepancy column."""
        lines = []
        
        # Header
        lines.append("# Signature Consistency Report")
        lines.append("")
        lines.append(f"**Source:** {report.source_file} | **Date:** {report.extraction_date[:10]}")
        lines.append(f"**Total Signatures:** {report.total_signatures} | **Unique Signers:** {report.unique_signers}")
        lines.append("")
        
        # Main summary table with discrepancies
        if report.signature_groups:
            lines.append("| Role | Count | Pages | Consistency | Discrepancies |")
            lines.append("|------|-------|-------|-------------|---------------|")
            
            for group in report.signature_groups:
                role = group.designation or group.identifier
                pages_str = ", ".join(str(p) for p in group.pages)
                consistency = f"{group.average_similarity:.2f}" if group.average_similarity > 0 else "-"
                
                # Format discrepancies
                if group.outliers:
                    discrepancies = "; ".join(
                        f"Page {o.page} (outlier: {o.avg_similarity:.2f})"
                        for o in group.outliers
                    )
                else:
                    discrepancies = "-"
                
                lines.append(
                    f"| {role} | {len(group.signatures)} | {pages_str} | "
                    f"{consistency} | {discrepancies} |"
                )
            
            lines.append("")
        else:
            lines.append("*No signatures detected in this document.*")
            lines.append("")
        
        # Discrepancy details section (only if there are outliers)
        has_outliers = any(group.outliers for group in report.signature_groups)
        if has_outliers:
            lines.append("---")
            lines.append("")
            lines.append("## Discrepancy Details")
            lines.append("")
            
            for group in report.signature_groups:
                if group.outliers:
                    role = group.designation or group.identifier
                    lines.append(f"### {role}")
                    lines.append("")
                    for outlier in group.outliers:
                        lines.append(f"- **Page {outlier.page}**: {outlier.reason}")
                    lines.append("")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _write_full_markdown_report(
        self,
        report: SignatureAnalysisReport,
        output_path: Path,
        output_dir: str
    ):
        """Write full detailed Markdown report (legacy mode)."""
        lines = []
        
        # Header
        lines.append("# Signature Analysis Report")
        lines.append("")
        lines.append(f"**Source:** {report.source_file} | **Date:** {report.extraction_date[:10]}")
        lines.append(f"**Signatures Detected:** {report.total_signatures} | **Unique Signers:** {report.unique_signers}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Summary by Signer
        lines.append("## Summary by Signer")
        lines.append("")
        
        if report.signature_groups:
            lines.append("| Signer | Designation | Occurrences | Pages | Consistency | Avg Score |")
            lines.append("|--------|-------------|-------------|-------|-------------|-----------|")
            
            for group in report.signature_groups:
                pages_str = ", ".join(str(p) for p in group.pages)
                avg_score = f"{group.average_similarity:.2f}" if group.average_similarity > 0 else "-"
                
                lines.append(
                    f"| {group.identifier} | {group.designation or '-'} | "
                    f"{len(group.signatures)} | {pages_str} | "
                    f"{group.internal_consistency} | {avg_score} |"
                )
            
            lines.append("")
        else:
            lines.append("*No signatures detected in this document.*")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Detailed Analysis per Signer
        for group in report.signature_groups:
            lines.append(f"## Detailed Analysis: {group.identifier}")
            if group.designation:
                lines.append(f"**Designation:** {group.designation}")
            lines.append("")
            
            # Signature Instances Table
            lines.append("### Signature Instances")
            lines.append("")
            lines.append("| ID | Page | Confidence | Visual Description |")
            lines.append("|----|------|------------|-------------------|")
            
            for sig in group.signatures:
                confidence_pct = f"{sig.confidence * 100:.0f}%"
                desc = sig.visual_description[:50] + "..." if len(sig.visual_description) > 50 else sig.visual_description
                lines.append(f"| {sig.signature_id} | {sig.page_number} | {confidence_pct} | {desc or '-'} |")
            
            lines.append("")
            
            # Comparison Results (if multiple signatures)
            if len(group.signatures) > 1:
                lines.append("### Comparison Results")
                lines.append("")
                
                # Get comparisons for this group
                group_comparisons = [
                    c for c in report.comparisons
                    if any(sig.signature_id in [c.signature_a_id, c.signature_b_id] 
                           for sig in group.signatures)
                ]
                
                if group_comparisons:
                    lines.append("| Pair | SSIM | ORB | Hash Dist | Score | Verdict |")
                    lines.append("|------|------|-----|-----------|-------|---------|")
                    
                    for comp in group_comparisons:
                        lines.append(
                            f"| {comp.signature_a_id} vs {comp.signature_b_id} | "
                            f"{comp.ssim_score:.2f} | {comp.orb_score:.2f} | "
                            f"{comp.hash_distance} | {comp.similarity_score:.2f} | "
                            f"{comp.match_verdict} |"
                        )
                    
                    lines.append("")
                
                # Overall verdict
                lines.append(f"**Overall Verdict:** {group.internal_consistency}")
                if group.average_similarity > 0:
                    lines.append(f" (Average similarity: {group.average_similarity:.2f})")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Summary Section
        lines.append("## Summary")
        lines.append("")
        
        if report.summary:
            for signer, summary_text in report.summary.items():
                lines.append(f"- **{signer}:** {summary_text}")
            lines.append("")
        
        # Signature Image References (if images exist)
        sig_dir = Path(output_dir) / self.config.signatures_dir
        if sig_dir.exists() and any(sig_dir.iterdir()):
            lines.append("---")
            lines.append("")
            lines.append("## Signature Images")
            lines.append("")
            lines.append(f"Signature images are saved in the `{self.config.signatures_dir}/` directory.")
            lines.append("")
            
            # List available images
            for sig in report.signatures:
                if sig.image_path:
                    rel_path = Path(sig.image_path).name
                    lines.append(f"- `{sig.signature_id}`: [{rel_path}]({self.config.signatures_dir}/{rel_path})")
            lines.append("")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def generate_signature_report(
    extraction: ExtractionResult,
    output_dir: str,
    comparator: Optional[SignatureComparator] = None,
    config: Optional[SignatureReportConfig] = None
) -> Dict[str, str]:
    """
    Convenience function to generate signature reports.
    
    Args:
        extraction: ExtractionResult from the pipeline
        output_dir: Directory to save reports
        comparator: Optional SignatureComparator with pre-computed comparisons
        config: Optional report configuration
        
    Returns:
        Dict with paths to generated files
    """
    generator = SignatureReportGenerator(config)
    
    # Get detailed comparisons from comparator if available
    detailed_comparisons = None
    if comparator:
        detailed_comparisons = comparator.get_detailed_comparisons()
    
    return generator.generate_report(
        extraction,
        output_dir,
        detailed_comparisons
    )
