#!/usr/bin/env python3
"""
Document Ingestion Tool Comparison - Main Orchestrator

This script orchestrates the complete comparison pipeline:
1. Load PDF document
2. Run all ingestion tools
3. Calculate quality metrics
4. Generate visualizations
5. Create HTML report
6. Display summary to console

Usage:
    python run_comparison.py --input <pdf_file> [--output-dir <directory>]

Example:
    python run_comparison.py --input ../docs/sample.pdf --output-dir results/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import our modules
from compare_ingestors import run_comparison
from quality_analyzer import analyze_quality, calculate_pairwise_similarity, get_summary_statistics, create_comparison_table
from visualize import generate_all_visualizations
from html_report import generate_html_report


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        📄 DOCUMENT INGESTION TOOL COMPARISON 📊                  ║
    ║                                                                   ║
    ║   Comparing 7 free PDF processing libraries:                     ║
    ║   • Docling          • Unstructured       • LayoutLM             ║
    ║   • PaddleOCR        • LlamaIndex         • PyMuPDF              ║
    ║   • PDFPlumber                                                    ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_input(pdf_path: str) -> Path:
    """
    Validate input PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a PDF
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if pdf_file.suffix.lower() != '.pdf':
        raise ValueError(f"File must be a PDF: {pdf_path}")
    
    return pdf_file


def save_json_results(results, metrics_list, summary_stats, output_dir):
    """
    Save results to JSON file
    
    Args:
        results: List of IngestionResult objects
        metrics_list: List of QualityMetrics objects
        summary_stats: Summary statistics dictionary
        output_dir: Output directory path
    """
    # Convert results to JSON-serializable format
    results_dict = []
    for r in results:
        results_dict.append({
            "tool_name": r.tool_name,
            "text": r.text[:1000] + "..." if len(r.text) > 1000 else r.text,  # Truncate for JSON
            "text_length": len(r.text),
            "metadata": r.metadata,
            "processing_time_ms": r.processing_time_ms,
            "structural_elements": {
                "headings": r.structural_elements.headings[:10],  # Sample only
                "num_headings": len(r.structural_elements.headings),
                "num_tables": len(r.structural_elements.tables),
                "num_lists": len(r.structural_elements.lists),
                "num_images": r.structural_elements.images,
                "image_paths": r.structural_elements.image_paths,
                "table_paths": r.structural_elements.table_paths,
            },
            "error": r.error,
            "success": r.success,
        })
    
    # Convert metrics to dictionary
    metrics_dict = []
    for m in metrics_list:
        metrics_dict.append({
            "tool_name": m.tool_name,
            "text_completeness_score": m.text_completeness_score,
            "structure_score": m.structure_score,
            "metadata_score": m.metadata_score,
            "word_count": m.word_count,
            "char_count": m.char_count,
            "num_headings": m.num_headings,
            "num_tables": m.num_tables,
            "num_lists": m.num_lists,
            "num_images": m.num_images,
            "metadata_fields": m.metadata_fields,
        })
    
    # Combine all data
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary_stats,
        "results": results_dict,
        "metrics": metrics_dict,
    }
    
    # Save to file
    output_file = output_dir / "comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ JSON results saved to: {output_file}")


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Compare document ingestion tools on PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comparison.py --input ../docs/sample.pdf
  python run_comparison.py --input report.pdf --output-dir my_results/
  
The script will create an output directory with:
  - comparison_results.json  : Raw results and metrics
  - report.html             : Interactive HTML report
  - *.png                   : Visualization charts
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to PDF file to analyze'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='comparison_output',
        help='Output directory for results (default: comparison_output)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    try:
        # Validate input
        print("🔍 Validating input...")
        pdf_file = validate_input(args.input)
        print(f"  ✓ PDF file: {pdf_file.name}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output directory: {output_dir}")
        
        # Step 1: Run all ingestion tools
        print("\n" + "="*70)
        print("STEP 1: Running Document Ingestion Tools")
        print("="*70)
        results = run_comparison(str(pdf_file), str(output_dir))
        
        # Count successes and failures
        num_success = sum(1 for r in results if r.success)
        num_failed = len(results) - num_success
        total_images = sum(r.structural_elements.images for r in results if r.success)
        total_tables = sum(len(r.structural_elements.tables) for r in results if r.success)
        print(f"\n✓ Completed: {num_success} succeeded, {num_failed} failed")
        print(f"✓ Extracted: {total_images} images, {total_tables} tables across all tools")
        
        # Step 2: Analyze quality
        print("\n" + "="*70)
        print("STEP 2: Analyzing Quality Metrics")
        print("="*70)
        metrics_list = analyze_quality(results)
        print("  ✓ Quality metrics calculated")
        
        # Calculate similarity matrix
        similarity_matrix = calculate_pairwise_similarity(results)
        print("  ✓ Pairwise similarity computed")
        
        # Get summary statistics
        summary_stats = get_summary_statistics(metrics_list)
        print("  ✓ Summary statistics generated")
        
        # Step 3: Generate visualizations
        print("\n" + "="*70)
        print("STEP 3: Creating Visualizations")
        print("="*70)
        chart_paths = generate_all_visualizations(
            results, 
            metrics_list, 
            similarity_matrix, 
            str(output_dir)
        )
        
        # Step 4: Generate HTML report
        print("\n" + "="*70)
        print("STEP 4: Generating HTML Report")
        print("="*70)
        html_path = generate_html_report(
            results,
            metrics_list,
            chart_paths,
            str(output_dir / "report.html"),
            pdf_file.name
        )
        
        # Step 5: Save JSON results
        print("\n" + "="*70)
        print("STEP 5: Saving Results")
        print("="*70)
        save_json_results(results, metrics_list, summary_stats, output_dir)
        
        # Display console summary
        print("\n" + "="*70)
        print("SUMMARY RESULTS")
        print("="*70)
        print(create_comparison_table(metrics_list))
        
        print("\n" + "="*70)
        print("KEY FINDINGS")
        print("="*70)
        print(f"  📊 Best Completeness:  {summary_stats.get('best_completeness_tool', 'N/A')}")
        print(f"  🏗️  Best Structure:     {summary_stats.get('best_structure_tool', 'N/A')}")
        print(f"  📋 Best Metadata:      {summary_stats.get('best_metadata_tool', 'N/A')}")
        print(f"  📈 Average Completeness: {summary_stats.get('avg_text_completeness', 0):.1f}%")
        print(f"  📈 Average Structure:    {summary_stats.get('avg_structure_score', 0):.1f}")
        print(f"  📈 Average Metadata:     {summary_stats.get('avg_metadata_score', 0):.1f}")
        
        # Show extraction statistics
        print("\n  📸 Image Extraction:")
        for r in results:
            if r.success and r.structural_elements.images > 0:
                print(f"     {r.tool_name:15s}: {r.structural_elements.images} images")
        
        print("\n  📋 Table Extraction:")
        for r in results:
            if r.success and len(r.structural_elements.tables) > 0:
                print(f"     {r.tool_name:15s}: {len(r.structural_elements.tables)} tables")
        
        print("\n" + "="*70)
        print("OUTPUT FILES")
        print("="*70)
        print(f"  📄 HTML Report:  {html_path}")
        print(f"  📊 JSON Results: {output_dir / 'comparison_results.json'}")
        print(f"  📈 Charts:       {len(chart_paths)} PNG files in {output_dir}")
        
        print("\n" + "="*70)
        print(f"✅ SUCCESS! Open {html_path} in your browser to view the report.")
        print("="*70 + "\n")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
