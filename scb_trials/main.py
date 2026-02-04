#!/usr/bin/env python3
"""
Document OCR Pipeline - Main CLI Entry Point

A comprehensive 4-stage document extraction pipeline for scanned PDFs.

Usage:
    python main.py <pdf_path> --provider <azure|vertex> [options]

Examples:
    python main.py document.pdf --provider azure
    python main.py document.pdf --provider vertex --dpi 400 --enhance --output-dir ./output
"""

import io
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def generate_run_id(pdf_path: str, provider: str) -> str:
    """
    Generate a unique run ID based on document name, provider, and timestamp.
    
    Format: {doc_name}_{provider}_{YYYYMMDD_HHMMSS}
    
    Args:
        pdf_path: Path to the PDF document
        provider: OCR provider name (azure/vertex)
        
    Returns:
        Unique run ID string
    """
    doc_name = Path(pdf_path).stem
    # Sanitize document name (remove special chars, limit length)
    doc_name = "".join(c if c.isalnum() or c == "_" else "_" for c in doc_name)[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{doc_name}_{provider}_{timestamp}"


def setup_logging(output_dir: Path, run_id: str, verbose: bool = False) -> logging.Logger:
    """
    Set up logging to both file and console.
    
    Args:
        output_dir: Directory to save log files
        run_id: Run ID for the log filename
        verbose: Enable verbose (DEBUG) logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("ocr_pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler - always DEBUG level for comprehensive logs
    log_file = output_dir / f"{run_id}.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - respects verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

from pipeline.preprocessor import Preprocessor, PreprocessConfig
from pipeline.extractor import Extractor, ExtractorConfig
from pipeline.postprocessor import Postprocessor
from pipeline.reviewer import Reviewer

def run_pipeline(
    pdf_path: str,
    provider: str,
    dpi: int = 300,
    enhance: bool = True,
    output_dir: str = "./output",
    save_artifacts: bool = False,
    signature_report: bool = True,
    verbose: bool = False,
    run_id: Optional[str] = None,
    show_console: bool = True,
    parallel: bool = True,  # Task 9.4: Enable parallel page processing by default
    max_workers: int = 3,   # Task 9.4: Max parallel workers (conservative for rate limits)
) -> Dict[str, Any]:
    """
    Run the OCR pipeline programmatically.
    
    Returns a dict with run_id, output_dir, output_paths, and elapsed_time.
    
    Task 9 Optimizations:
    - parallel: Enable parallel page processing (9.4)
    - max_workers: Control parallelism (default 3 for rate limits)
    """
    start_time = time.time()
    console = Console() if show_console else Console(file=io.StringIO())
    
    # Generate unique run ID
    run_id = run_id or generate_run_id(pdf_path, provider)
    
    # Create run-specific output directory
    base_output_path = Path(output_dir)
    output_path = base_output_path / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_path, run_id, verbose)
    logger.info(f"=" * 60)
    logger.info(f"OCR Pipeline Run Started")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"=" * 60)
    logger.info(f"Input PDF: {pdf_path}")
    logger.info(f"Provider: {provider}")
    logger.info(f"DPI: {dpi}")
    logger.info(f"Image Enhancement: {enhance}")
    logger.info(f"Output Directory: {output_path}")
    logger.info(f"Signature Report: {signature_report}")
    logger.info(f"Save Artifacts: {save_artifacts}")
    logger.info(f"Parallel Processing: {parallel} (max_workers={max_workers})")
    
    # Display header
    if show_console:
        console.print(Panel.fit(
            "[bold blue]Document OCR Pipeline[/bold blue]\n"
            f"Processing: {pdf_path}\n"
            f"Provider: {provider.upper()}\n"
            f"Run ID: [cyan]{run_id}[/cyan]",
            border_style="blue"
        ))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            # Stage 1: Preprocessing
            logger.info("-" * 50)
            logger.info("STAGE 1: PREPROCESSING")
            logger.info("-" * 50)
            task1 = progress.add_task("[cyan]Stage 1: Preprocessing...", total=100)
            
            preprocess_config = PreprocessConfig(
                dpi=dpi,
                enhance=enhance,
                save_intermediates=save_artifacts,
                output_dir=str(output_path / "artifacts") if save_artifacts else None
            )
            
            preprocessor = Preprocessor(config=preprocess_config)
            
            logger.info(f"Converting PDF at {dpi} DPI with enhancement={enhance}")
            if verbose and show_console:
                console.print(f"  Converting PDF at {dpi} DPI...")
            
            pages = preprocessor.process_pdf(pdf_path)
            progress.update(task1, completed=100)
            
            logger.info(f"Successfully converted {len(pages)} pages")
            if verbose and show_console:
                console.print(f"  [green]✓[/green] Converted {len(pages)} pages")
            
            # Stage 2: Extraction
            logger.info("-" * 50)
            logger.info("STAGE 2: EXTRACTION (Document Intelligence + LLM)")
            logger.info("-" * 50)
            task2 = progress.add_task("[cyan]Stage 2: Extraction...", total=100)
            
            # Initialize provider
            logger.info(f"Initializing {provider.upper()} provider...")
            ocr_provider = get_provider(provider)
            logger.info(f"Provider ready: {ocr_provider.name}, LLM-only mode: {getattr(ocr_provider, 'llm_only_mode', 'N/A')}")
            
            extractor_config = ExtractorConfig(
                extract_tables=True,
                extract_signatures=True,
                detect_redactions=True,
                use_llm_for_text=True,
                use_llm_for_tables=True,
                save_signatures=save_artifacts or signature_report,  # Save if either is enabled
                output_dir=str(output_path) if (save_artifacts or signature_report) else None,
                # Enhanced signature detection settings
                signature_detection_method="hybrid",  # Use both CV and LLM
                signature_page_region="lower_half",
                # Task 9 Performance Optimizations (never sacrifice accuracy)
                skip_llm_scan_if_found=True,  # 9.2: Skip redundant LLM scan
                batch_signature_validation=True,  # 9.1: Batch CV candidates
                batch_table_enhancement=True,  # 9.5: Batch tables
                skip_text_overlap_regions=True,  # 9.3: Filter text overlaps
            )
            logger.info(f"Extractor config: signature_method={extractor_config.signature_detection_method}, save_signatures={extractor_config.save_signatures}")
            logger.info(f"Optimizations: batch_signatures={extractor_config.batch_signature_validation}, batch_tables={extractor_config.batch_table_enhancement}, skip_llm_if_found={extractor_config.skip_llm_scan_if_found}")
            
            extractor = Extractor(provider=ocr_provider, config=extractor_config)
            
            logger.info(f"Starting extraction of {len(pages)} pages (parallel={parallel})...")
            if verbose and show_console:
                console.print(f"  Extracting content using {provider.upper()}...")
            
            extraction_result = extractor.extract_document(
                pages, 
                pdf_path,
                parallel=parallel,
                max_workers=max_workers
            )
            progress.update(task2, completed=100)
            
            table_count = sum(len(p.tables) for p in extraction_result.pages)
            sig_count = sum(len(p.signatures) for p in extraction_result.pages)
            text_blocks = sum(len(p.text_blocks) for p in extraction_result.pages)
            redaction_count = sum(len(p.redactions) for p in extraction_result.pages)
            
            logger.info(f"Extraction complete:")
            logger.info(f"  - Text blocks: {text_blocks}")
            logger.info(f"  - Tables: {table_count}")
            logger.info(f"  - Signatures detected: {sig_count}")
            logger.info(f"  - Redactions: {redaction_count}")
            logger.info(f"  - Overall confidence: {extraction_result.overall_confidence:.2%}")
            
            # Task 9.6: Log timing summary
            if extraction_result.metadata.get('timing_summary'):
                timing_summary = extraction_result.metadata['timing_summary']
                logger.info(f"  - Extraction time: {extraction_result.metadata.get('extraction_time_seconds', 0):.1f}s")
                logger.info(f"  - Timing breakdown:")
                logger.info(f"      DI Layout: {timing_summary.get('di_layout_ms', 0):.0f}ms")
                logger.info(f"      CV Detection: {timing_summary.get('cv_signature_detection_ms', 0):.0f}ms")
                logger.info(f"      Signature Validation: {timing_summary.get('signature_validation_ms', 0):.0f}ms ({timing_summary.get('signature_validation_count', 0)} candidates)")
                logger.info(f"      LLM Signature Scan: {timing_summary.get('llm_signature_scan_ms', 0):.0f}ms ({timing_summary.get('llm_scans_skipped', 0)} skipped)")
                logger.info(f"      Table Enhancement: {timing_summary.get('table_enhancement_ms', 0):.0f}ms ({timing_summary.get('table_enhancement_count', 0)} tables)")
            
            # Log individual page details
            for page in extraction_result.pages:
                logger.debug(f"  Page {page.page_number}: {len(page.text_blocks)} text, {len(page.tables)} tables, {len(page.signatures)} sigs, confidence={page.confidence:.2%}")
            
            if verbose and show_console:
                console.print(f"  [green]✓[/green] Found {table_count} tables, {sig_count} signatures")
            
            # Stage 3: Post-processing
            logger.info("-" * 50)
            logger.info("STAGE 3: POST-PROCESSING (Signature Analysis)")
            logger.info("-" * 50)
            task3 = progress.add_task("[cyan]Stage 3: Post-processing...", total=100)
            
            logger.info("Merging content blocks and comparing signatures...")
            postprocessor = Postprocessor()
            processed_doc = postprocessor.process(extraction_result)
            
            # Get the signature comparator for detailed comparison data
            signature_comparator = postprocessor.signature_comparator
            
            progress.update(task3, completed=100)
            
            logger.info(f"Post-processing complete:")
            logger.info(f"  - Content blocks: {len(processed_doc.content_blocks)}")
            logger.info(f"  - Financial values: {len(processed_doc.financial_values)}")
            logger.info(f"  - Signature groups: {len(processed_doc.signatures)}")
            
            # Log signature comparison results
            for sig in processed_doc.signatures:
                logger.info(f"  Signature group '{sig.name}': {sig.comparison_result}")
                logger.info(f"    - Designation: {sig.designation}")
                logger.info(f"    - Pages: {sig.pages}")
                if sig.similarity_scores:
                    logger.info(f"    - Similarity scores: {[f'{s:.3f}' for s in sig.similarity_scores]}")
                if sig.notes:
                    logger.info(f"    - Notes: {sig.notes}")
            
            if verbose and show_console:
                console.print(f"  [green]✓[/green] Processed {len(processed_doc.content_blocks)} content blocks")
                console.print(f"  [green]✓[/green] Found {len(processed_doc.financial_values)} financial values")
                if signature_report:
                    console.print(f"  [green]✓[/green] Signature comparison completed")
            
            # Stage 4: Review and Output
            logger.info("-" * 50)
            logger.info("STAGE 4: REVIEW & OUTPUT")
            logger.info("-" * 50)
            task4 = progress.add_task("[cyan]Stage 4: Review & Output...", total=100)
            
            reviewer = Reviewer(output_dir=str(output_path))
            confidence_report = reviewer.review(processed_doc)
            
            logger.info(f"Confidence report generated:")
            logger.info(f"  - Overall score: {confidence_report.overall_score:.2%}")
            logger.info(f"  - Needs review: {confidence_report.needs_review()}")
            logger.info(f"  - Warnings: {len(confidence_report.warnings)}")
            for warning in confidence_report.warnings:
                logger.warning(f"  {warning}")
            
            # Generate outputs based on format option
            # Pass extraction and comparator for signature report generation
            logger.info("Generating output files...")
            output_paths = reviewer.save_outputs(
                processed_doc,
                confidence_report,
                extraction=extraction_result,
                comparator=signature_comparator,
                generate_signature_report=signature_report
            )
            progress.update(task4, completed=100)
            
            logger.info("Output files saved:")
            for output_type, path in output_paths.items():
                logger.info(f"  - {output_type}: {path}")
        
        # Display results
        elapsed_time = time.time() - start_time
        
        logger.info("-" * 50)
        logger.info("RUN COMPLETE")
        logger.info("-" * 50)
        logger.info(f"Total processing time: {elapsed_time:.2f}s")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Output directory: {output_path}")
        
        if show_console:
            console.print()
            console.print(Panel.fit(
                "[bold green]Extraction Complete![/bold green]",
                border_style="green"
            ))
            
            # Summary table
            summary_table = Table(title="Extraction Summary", show_header=True)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white")
            
            summary_table.add_row("Run ID", run_id)
            summary_table.add_row("Total Pages", str(processed_doc.total_pages))
            summary_table.add_row("Tables Extracted", str(processed_doc.table_count))
            summary_table.add_row("Signatures Found", str(len(processed_doc.signatures)))
            summary_table.add_row("Financial Values", str(len(processed_doc.financial_values)))
            summary_table.add_row("Redactions", str(processed_doc.redaction_count))
            summary_table.add_row("Overall Confidence", f"{processed_doc.overall_confidence:.1%}")
            summary_table.add_row("Processing Time", f"{elapsed_time:.2f}s")
            
            console.print(summary_table)
            
            # Signature analysis
            if processed_doc.signatures:
                console.print()
                sig_table = Table(title="Signature Analysis", show_header=True)
                sig_table.add_column("Name", style="cyan")
                sig_table.add_column("Designation", style="white")
                sig_table.add_column("Pages", style="white")
                sig_table.add_column("Comparison", style="white")
                
                for sig in processed_doc.signatures:
                    comparison_style = "green" if sig.comparison_result == "Consistent" else "red" if sig.comparison_result == "Discrepancy" else "yellow"
                    sig_table.add_row(
                        sig.name or "Unknown",
                        sig.designation or "-",
                        ", ".join(str(p) for p in sig.pages),
                        f"[{comparison_style}]{sig.comparison_result}[/{comparison_style}]"
                    )
                
                console.print(sig_table)
            
            # Warnings
            if confidence_report.warnings:
                console.print()
                console.print("[yellow]Warnings:[/yellow]")
                for warning in confidence_report.warnings:
                    console.print(f"  [yellow]![/yellow] {warning}")
            
            # Output files
            console.print()
            console.print(f"[bold]Output Directory:[/bold] {output_path}")
            console.print("[bold]Output Files:[/bold]")
            for output_type, path in output_paths.items():
                console.print(f"  • {output_type}: {Path(path).name}")
            
            # Log file location
            console.print(f"  • log: {run_id}.log")
            
            # Review needed?
            if confidence_report.needs_review():
                console.print()
                console.print(Panel.fit(
                    "[bold yellow]Manual Review Recommended[/bold yellow]\n"
                    "This document has been flagged for review due to:\n"
                    "• Low confidence regions or signature discrepancies",
                    border_style="yellow"
                ))
        
        return {
            "run_id": run_id,
            "output_dir": str(output_path),
            "output_paths": output_paths,
            "elapsed_time": elapsed_time,
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        if show_console:
            console.print(f"[red]Error:[/red] {str(e)}")
            console.print(f"[red]Check log file for details:[/red] {output_path / f'{run_id}.log'}")
            if verbose:
                console.print_exception()
        raise

def get_provider(provider_name: str):
    """
    Get the appropriate OCR provider.
    
    Args:
        provider_name: 'azure' or 'vertex'
        
    Returns:
        OCRProvider instance
    """
    if provider_name.lower() == 'azure':
        from providers.azure_provider import AzureProvider
        return AzureProvider()
    elif provider_name.lower() == 'vertex':
        from providers.vertex_provider import VertexProvider
        return VertexProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Use 'azure' or 'vertex'.")


@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option(
    '--provider', '-p',
    type=click.Choice(['azure', 'vertex'], case_sensitive=False),
    required=True,
    help='OCR provider to use (azure or vertex)'
)
@click.option(
    '--dpi',
    type=int,
    default=300,
    help='DPI for PDF to image conversion (default: 300)'
)
@click.option(
    '--enhance/--no-enhance',
    default=True,
    help='Apply image enhancement (default: enabled)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(),
    default='./output',
    help='Output directory for results (default: ./output)'
)
@click.option(
    '--save-artifacts/--no-artifacts',
    default=False,
    help='Save intermediate artifacts (images, signatures)'
)
@click.option(
    '--signature-report/--no-signature-report',
    default=True,
    help='Generate detailed signature analysis report (default: enabled)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Verbose output'
)
@click.option(
    '--parallel/--no-parallel',
    default=True,
    help='Enable parallel page processing (default: enabled)'
)
@click.option(
    '--max-workers',
    type=int,
    default=3,
    help='Maximum parallel workers (default: 3)'
)
def main(
    pdf_path: str,
    provider: str,
    dpi: int,
    enhance: bool,
    output_dir: str,
    save_artifacts: bool,
    signature_report: bool,
    verbose: bool,
    parallel: bool,
    max_workers: int
):
    """
    Extract content from a scanned PDF document.
    
    PDF_PATH: Path to the PDF file to process
    """
    try:
        run_pipeline(
            pdf_path=pdf_path,
            provider=provider,
            dpi=dpi,
            enhance=enhance,
            output_dir=output_dir,
            save_artifacts=save_artifacts,
            signature_report=signature_report,
            verbose=verbose,
            show_console=True,
            parallel=parallel,
            max_workers=max_workers,
        )
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
