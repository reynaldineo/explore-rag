"""
HTML Report Generator for Document Ingestion Comparison

Generates an interactive HTML report with:
- Summary table of all metrics
- Side-by-side text previews
- Structure comparison section
- Embedded visualization charts
- Collapsible sections for full text
"""

import base64
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 for embedding in HTML
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string
    """
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        return f"Error loading image: {str(e)}"


def generate_html_header(pdf_name: str) -> str:
    """Generate HTML header with CSS styling"""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Ingestion Comparison - {pdf_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        
        h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        
        h2 {{
            color: #764ba2;
            font-size: 1.8em;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        
        h3 {{
            color: #667eea;
            font-size: 1.3em;
            margin: 20px 0 10px 0;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
            border-radius: 8px;
        }}
        
        .summary-table thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .summary-table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .summary-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        .summary-table tbody tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .summary-table tbody tr:last-child td {{
            border-bottom: none;
        }}
        
        .tool-name {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .success {{
            color: #2ecc71;
            font-weight: 600;
        }}
        
        .error {{
            color: #e74c3c;
            font-weight: 600;
        }}
        
        .score-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .score-high {{
            background-color: #d4edda;
            color: #155724;
        }}
        
        .score-medium {{
            background-color: #fff3cd;
            color: #856404;
        }}
        
        .score-low {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        .text-preview {{
            background-color: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .text-preview pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }}
        
        .collapsible {{
            background-color: #667eea;
            color: white;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1em;
            font-weight: 600;
            border-radius: 5px;
            margin: 10px 0;
            transition: background-color 0.3s;
        }}
        
        .collapsible:hover {{
            background-color: #5568d3;
        }}
        
        .collapsible:after {{
            content: '\\002B';
            color: white;
            font-weight: bold;
            float: right;
            margin-left: 5px;
        }}
        
        .collapsible.active:after {{
            content: "\\2212";
        }}
        
        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background-color: #f9f9f9;
            border-radius: 0 0 5px 5px;
        }}
        
        .content-inner {{
            padding: 20px;
        }}
        
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .card {{
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .card h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .metadata-item {{
            margin: 5px 0;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metadata-item:last-child {{
            border-bottom: none;
        }}
        
        .metadata-key {{
            font-weight: 600;
            color: #666;
        }}
        
        .metadata-value {{
            color: #333;
        }}
        
        .structure-list {{
            list-style: none;
            padding: 0;
        }}
        
        .structure-list li {{
            padding: 8px;
            margin: 5px 0;
            background-color: white;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }}
        
        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            color: #999;
            font-size: 0.9em;
        }}
        
        .best-indicator {{
            background-color: #ffd700;
            color: #333;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: 600;
            margin-left: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📄 Document Ingestion Tool Comparison</h1>
            <p class="subtitle">Comprehensive analysis of {pdf_name}</p>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""


def generate_summary_table(results: List[Any], metrics_list: List[Any]) -> str:
    """Generate summary table HTML"""
    
    # Find best performers
    best_time = min([r.processing_time_ms for r in results if r.success], default=0)
    best_completeness = max([m.text_completeness_score for m in metrics_list], default=0)
    best_structure = max([m.structure_score for m in metrics_list], default=0)
    best_metadata = max([m.metadata_score for m in metrics_list], default=0)
    
    html = """
        <section id="summary">
            <h2>📊 Summary Table</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Tool</th>
                        <th>Status</th>
                        <th>Time (ms)</th>
                        <th>Words</th>
                        <th>Completeness</th>
                        <th>Structure</th>
                        <th>Metadata</th>
                        <th>Elements (H/T/L/I)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for result, metrics in zip(results, metrics_list):
        status = '<span class="success">✓ Success</span>' if result.success else '<span class="error">✗ Failed</span>'
        
        # Score badges
        def get_score_class(score):
            if score >= 80:
                return 'score-high'
            elif score >= 50:
                return 'score-medium'
            else:
                return 'score-low'
        
        completeness_badge = f'<span class="score-badge {get_score_class(metrics.text_completeness_score)}">{metrics.text_completeness_score:.1f}%</span>'
        structure_badge = f'<span class="score-badge {get_score_class(metrics.structure_score)}">{metrics.structure_score:.1f}</span>'
        metadata_badge = f'<span class="score-badge {get_score_class(metrics.metadata_score)}">{metrics.metadata_score:.1f}</span>'
        
        # Add best indicators
        if result.success and result.processing_time_ms == best_time:
            time_str = f'{result.processing_time_ms:.0f}<span class="best-indicator">FASTEST</span>'
        else:
            time_str = f'{result.processing_time_ms:.0f}'
            
        if metrics.text_completeness_score == best_completeness:
            completeness_badge += '<span class="best-indicator">BEST</span>'
        if metrics.structure_score == best_structure:
            structure_badge += '<span class="best-indicator">BEST</span>'
        if metrics.metadata_score == best_metadata:
            metadata_badge += '<span class="best-indicator">BEST</span>'
        
        elements = f"{metrics.num_headings}/{metrics.num_tables}/{metrics.num_lists}/{metrics.num_images}"
        
        html += f"""
                    <tr>
                        <td class="tool-name">{result.tool_name}</td>
                        <td>{status}</td>
                        <td>{time_str}</td>
                        <td>{metrics.word_count:,}</td>
                        <td>{completeness_badge}</td>
                        <td>{structure_badge}</td>
                        <td>{metadata_badge}</td>
                        <td>{elements}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
            <p style="margin-top: 10px; color: #666; font-size: 0.9em;">
                <strong>H/T/L/I:</strong> Headings / Tables / Lists / Images
            </p>
        </section>
"""
    
    return html


def generate_charts_section(chart_paths: Dict[str, str]) -> str:
    """Generate charts section with embedded images"""
    html = """
        <section id="charts">
            <h2>📈 Visual Comparisons</h2>
"""
    
    chart_titles = {
        'processing_time': 'Processing Speed',
        'text_completeness': 'Text Extraction Completeness',
        'structure_scores': 'Structure Detection',
        'similarity_heatmap': 'Tool Similarity Matrix',
        'overall_scores': 'Overall Quality Scores'
    }
    
    for key, path in chart_paths.items():
        if Path(path).exists():
            img_data = encode_image_to_base64(path)
            title = chart_titles.get(key, key.replace('_', ' ').title())
            
            html += f"""
            <div class="chart-container">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{img_data}" alt="{title}">
            </div>
"""
    
    html += """
        </section>
"""
    
    return html


def generate_text_preview_section(results: List[Any]) -> str:
    """Generate text preview section with collapsible full text"""
    html = """
        <section id="text-previews">
            <h2>📝 Text Extraction Previews</h2>
            <p style="color: #666; margin-bottom: 20px;">
                First 500 characters from each tool. Click to expand for full text.
            </p>
"""
    
    for result in results:
        if result.success and result.text:
            preview = result.text[:500].replace('<', '&lt;').replace('>', '&gt;')
            full_text = result.text.replace('<', '&lt;').replace('>', '&gt;')
            
            html += f"""
            <div style="margin: 20px 0;">
                <h3>{result.tool_name}</h3>
                <div class="text-preview">
                    <pre>{preview}...</pre>
                </div>
                <button class="collapsible">Show Full Text ({len(result.text):,} characters)</button>
                <div class="content">
                    <div class="content-inner">
                        <pre>{full_text}</pre>
                    </div>
                </div>
            </div>
"""
        else:
            html += f"""
            <div style="margin: 20px 0;">
                <h3>{result.tool_name}</h3>
                <div class="text-preview">
                    <span class="error">Failed to extract text: {result.error}</span>
                </div>
            </div>
"""
    
    html += """
        </section>
"""
    
    return html


def generate_structure_section(results: List[Any], metrics_list: List[Any]) -> str:
    """Generate structure comparison section"""
    html = """
        <section id="structure">
            <h2>🏗️ Structure Detection Details</h2>
            <div class="grid">
"""
    
    for result, metrics in zip(results, metrics_list):
        html += f"""
            <div class="card">
                <h4>{result.tool_name}</h4>
"""
        
        if result.success:
            html += f"""
                <div class="metadata-item">
                    <span class="metadata-key">Headings:</span>
                    <span class="metadata-value">{metrics.num_headings}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-key">Tables:</span>
                    <span class="metadata-value">{metrics.num_tables}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-key">Lists:</span>
                    <span class="metadata-value">{metrics.num_lists}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-key">Images:</span>
                    <span class="metadata-value">{metrics.num_images}</span>
                </div>
"""
            
            # Show sample headings
            if result.structural_elements.headings:
                sample_headings = result.structural_elements.headings[:3]
                html += """
                <div style="margin-top: 15px;">
                    <strong style="color: #667eea;">Sample Headings:</strong>
                    <ul class="structure-list">
"""
                for heading in sample_headings:
                    heading_clean = heading.replace('<', '&lt;').replace('>', '&gt;')[:100]
                    html += f"                        <li>{heading_clean}</li>\n"
                
                html += """
                    </ul>
                </div>
"""
        else:
            html += f"""
                <span class="error">Failed: {result.error}</span>
"""
        
        html += """
            </div>
"""
    
    html += """
            </div>
        </section>
"""
    
    return html


def generate_metadata_section(results: List[Any]) -> str:
    """Generate metadata comparison section"""
    html = """
        <section id="metadata">
            <h2>📋 Metadata Extraction</h2>
            <div class="grid">
"""
    
    for result in results:
        html += f"""
            <div class="card">
                <h4>{result.tool_name}</h4>
"""
        
        if result.success and result.metadata:
            for key, value in result.metadata.items():
                if value and str(value).strip():
                    value_str = str(value)[:100]
                    html += f"""
                <div class="metadata-item">
                    <span class="metadata-key">{key}:</span>
                    <span class="metadata-value">{value_str}</span>
                </div>
"""
        else:
            html += """
                <p style="color: #999;">No metadata extracted</p>
"""
        
        html += """
            </div>
"""
    
    html += """
            </div>
        </section>
"""
    
    return html


def generate_html_footer() -> str:
    """Generate HTML footer with JavaScript"""
    return """
        <footer>
            <p>Generated by Document Ingestion Comparison Tool</p>
            <p>Built with Python • Matplotlib • scikit-learn</p>
        </footer>
    </div>
    
    <script>
        // Add collapsible functionality
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }
        
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>
"""


def generate_html_report(results: List[Any], metrics_list: List[Any], 
                        chart_paths: Dict[str, str], output_path: str, pdf_name: str) -> str:
    """
    Generate complete HTML report
    
    Args:
        results: List of IngestionResult objects
        metrics_list: List of QualityMetrics objects
        chart_paths: Dictionary of chart file paths
        output_path: Path to save HTML file
        pdf_name: Name of the PDF being analyzed
        
    Returns:
        Path to generated HTML file
    """
    print("Generating HTML report...")
    
    # Build HTML document
    html = generate_html_header(pdf_name)
    html += generate_summary_table(results, metrics_list)
    html += generate_charts_section(chart_paths)
    html += generate_text_preview_section(results)
    html += generate_structure_section(results, metrics_list)
    html += generate_metadata_section(results)
    html += generate_html_footer()
    
    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ HTML report saved to: {output_file}")
    
    return str(output_file)


if __name__ == "__main__":
    print("HTML Report Generator Module")
    print("This module is meant to be imported and used by other scripts.")
    print("\nMain function:")
    print("  - generate_html_report(results, metrics_list, chart_paths, output_path, pdf_name)")
