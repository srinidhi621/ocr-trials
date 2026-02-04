#!/usr/bin/env python3
"""
Test Suite for Document OCR Pipeline
Tests the pipeline components without requiring actual API credentials.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test provider imports
        from providers.base import (
            OCRProvider,
            LayoutResult,
            RegionType,
            Region,
            Table,
            TableCell,
            SignatureBlock,
            BoundingBox,
            PageExtraction,
            ExtractionResult,
        )
        print("  ✓ providers.base")
        
        from providers.azure_provider import AzureProvider
        print("  ✓ providers.azure_provider")
        
        from providers.vertex_provider import VertexProvider
        print("  ✓ providers.vertex_provider")
        
        # Test pipeline imports
        from pipeline.preprocessor import Preprocessor, PreprocessConfig
        print("  ✓ pipeline.preprocessor")
        
        from pipeline.extractor import Extractor, ExtractorConfig
        print("  ✓ pipeline.extractor")
        
        from pipeline.postprocessor import (
            Postprocessor,
            SignatureComparator,
            FinancialValidator,
            ProcessedDocument,
        )
        print("  ✓ pipeline.postprocessor")
        
        from pipeline.reviewer import Reviewer, ConfidenceReport
        print("  ✓ pipeline.reviewer")
        
        # Test utility imports
        from utils.image_utils import load_image, save_image, crop_region
        print("  ✓ utils.image_utils")
        
        from utils.financial_utils import (
            extract_financial_values,
            normalize_financial_value,
            validate_financial_format,
        )
        print("  ✓ utils.financial_utils")
        
        print("\n✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False


def test_data_structures():
    """Test data structure creation and manipulation."""
    print("\nTesting data structures...")
    
    from providers.base import (
        BoundingBox,
        TableCell,
        Table,
        SignatureBlock,
        Region,
        RegionType,
        LayoutResult,
    )
    
    # Test BoundingBox
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    assert bbox.to_tuple() == (10, 20, 100, 50)
    assert bbox.to_xyxy() == (10, 20, 110, 70)
    print("  ✓ BoundingBox")
    
    # Test TableCell
    cell = TableCell(row_index=0, column_index=0, content="Header", row_span=1, column_span=2)
    assert cell.content == "Header"
    assert cell.column_span == 2
    print("  ✓ TableCell")
    
    # Test Table
    cells = [
        TableCell(0, 0, "Name"),
        TableCell(0, 1, "Value"),
        TableCell(1, 0, "Item 1"),
        TableCell(1, 1, "100"),
    ]
    table = Table(cells=cells, row_count=2, column_count=2)
    grid = table.to_grid()
    assert grid[0][0] == "Name"
    assert grid[1][1] == "100"
    markdown = table.to_markdown()
    assert "| Name | Value |" in markdown
    print("  ✓ Table")
    
    # Test SignatureBlock
    sig = SignatureBlock(
        name="John Smith",
        designation="Managing Director",
        page_number=1
    )
    assert sig.name == "John Smith"
    print("  ✓ SignatureBlock")
    
    # Test Region
    region = Region(
        region_type=RegionType.TEXT_BLOCK,
        bbox=bbox,
        content="Sample text"
    )
    assert region.region_type == RegionType.TEXT_BLOCK
    print("  ✓ Region")
    
    # Test LayoutResult
    layout = LayoutResult(
        page_number=1,
        regions=[region],
        width=800,
        height=1200,
        raw_text="Sample text"
    )
    text_blocks = layout.get_regions_by_type(RegionType.TEXT_BLOCK)
    assert len(text_blocks) == 1
    print("  ✓ LayoutResult")
    
    print("\n✓ All data structures working!")
    return True


def test_financial_utils():
    """Test financial value extraction and validation."""
    print("\nTesting financial utilities...")
    
    from utils.financial_utils import (
        extract_financial_values,
        normalize_financial_value,
        validate_financial_format,
        compare_financial_values,
    )
    
    # Test extraction
    text = "The facility amount is INR 500,000,000.00 with interest rate of 12.5% p.a."
    values = extract_financial_values(text)
    assert len(values) >= 2
    print(f"  ✓ Extracted {len(values)} financial values from text")
    
    # Test normalization
    amount, currency = normalize_financial_value("INR 500,000,000.00")
    assert amount == 500_000_000.0
    assert currency == "INR"
    print("  ✓ Normalized currency value")
    
    amount, currency = normalize_financial_value("50 Crore")
    assert amount == 500_000_000.0
    print("  ✓ Normalized unit value (Crore)")
    
    # Test validation
    assert validate_financial_format("INR 500,000.00")
    assert validate_financial_format("$1,234.56")
    assert validate_financial_format("15.5%")
    print("  ✓ Format validation working")
    
    # Test comparison
    assert compare_financial_values("INR 50 Crore", "INR 500,000,000")
    assert not compare_financial_values("INR 50 Crore", "USD 500,000,000")
    print("  ✓ Value comparison working")
    
    print("\n✓ All financial utilities working!")
    return True


def test_postprocessor_components():
    """Test postprocessor components."""
    print("\nTesting postprocessor components...")
    
    from pipeline.postprocessor import SignatureComparator, FinancialValidator
    from providers.base import SignatureBlock
    
    # Test SignatureComparator
    comparator = SignatureComparator()
    
    sig1 = SignatureBlock(name="John Smith", designation="MD", page_number=1, visual_description="cursive with loop")
    sig2 = SignatureBlock(name="John Smith", designation="MD", page_number=5, visual_description="cursive with loop")
    
    comparator.add_signature(sig1)
    comparator.add_signature(sig2)
    
    results = comparator.compare_all()
    assert len(results) == 1
    assert results[0].comparison_result == "Consistent"
    print("  ✓ SignatureComparator working")
    
    # Test FinancialValidator
    validator = FinancialValidator()
    
    text = "Total amount: INR 100,000,000.00. Interest: 8.5%"
    values = validator.extract_financial_values(text, page=1)
    assert len(values) >= 2
    print("  ✓ FinancialValidator working")
    
    print("\n✓ All postprocessor components working!")
    return True


def test_preprocessor_config():
    """Test preprocessor configuration."""
    print("\nTesting preprocessor configuration...")
    
    from pipeline.preprocessor import PreprocessConfig, Preprocessor
    
    # Test default config
    config = PreprocessConfig()
    assert config.dpi == 300
    assert config.enhance == True
    print("  ✓ Default config")
    
    # Test custom config
    config = PreprocessConfig(dpi=400, enhance=False, deskew=False)
    assert config.dpi == 400
    assert config.enhance == False
    print("  ✓ Custom config")
    
    # Test preprocessor initialization
    preprocessor = Preprocessor(config=config)
    assert preprocessor.config.dpi == 400
    print("  ✓ Preprocessor initialization")
    
    print("\n✓ Preprocessor configuration working!")
    return True


def test_reviewer():
    """Test reviewer functionality."""
    print("\nTesting reviewer...")
    
    from pipeline.reviewer import Reviewer, ConfidenceReport, LowConfidenceRegion
    from pipeline.postprocessor import ProcessedDocument, ContentBlock, SignatureComparison, FinancialValue
    
    # Create mock document
    content_blocks = [
        ContentBlock(block_type="text", content="Sample text", page=1, position=(0, 0)),
        ContentBlock(block_type="table", content="| A | B |\n|---|---|\n| 1 | 2 |", page=1, position=(100, 0)),
    ]
    
    signatures = [
        SignatureComparison(
            name="John Smith",
            designation="MD",
            pages=[1, 5],
            comparison_result="Consistent"
        )
    ]
    
    financial_values = [
        FinancialValue(value="INR 100,000,000.00", page=1, value_type="currency")
    ]
    
    doc = ProcessedDocument(
        source_file="test.pdf",
        provider="azure",
        total_pages=5,
        content_blocks=content_blocks,
        signatures=signatures,
        financial_values=financial_values,
        redaction_count=1,
        table_count=1,
        overall_confidence=0.95
    )
    
    # Test reviewer
    reviewer = Reviewer(output_dir="./test_output")
    report = reviewer.review(doc)
    
    assert report.overall_score == 0.95
    assert report.tables_extracted == 1
    print("  ✓ Review generation")
    
    # Test markdown generation
    markdown = reviewer.generate_markdown(doc, report)
    assert "## Signature Analysis" in markdown
    assert "John Smith" in markdown
    print("  ✓ Markdown generation")
    
    # Test JSON generation
    json_output = reviewer.generate_json(doc, report)
    assert json_output["metadata"]["source_file"] == "test.pdf"
    assert len(json_output["signatures"]) == 1
    print("  ✓ JSON generation")
    
    print("\n✓ Reviewer working!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Document OCR Pipeline - Component Tests")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        test_imports,
        test_data_structures,
        test_financial_utils,
        test_postprocessor_components,
        test_preprocessor_config,
        test_reviewer,
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
