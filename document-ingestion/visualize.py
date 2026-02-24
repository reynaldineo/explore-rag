"""
Visualization Module for Document Ingestion Comparison

Creates matplotlib charts to visualize comparison metrics:
- Processing time comparison (bar chart)
- Text completeness comparison (bar chart)
- Structure detection comparison (grouped bar chart)
- Similarity heatmap (pairwise tool similarity)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns


def setup_plot_style():
    """Configure matplotlib style for consistent, professional plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_processing_time(metrics_list: List[Any], results: List[Any], output_path: str) -> str:
    """
    Create horizontal bar chart of processing times
    
    Args:
        metrics_list: List of QualityMetrics objects
        results: List of IngestionResult objects (for timing data)
        output_path: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    setup_plot_style()
    
    # Extract data
    tool_names = [m.tool_name for m in metrics_list]
    times = [r.processing_time_ms for r in results]
    
    # Create color map (green for fast, red for slow)
    max_time = max(times)
    colors = plt.cm.RdYlGn_r(np.array(times) / max_time)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(tool_names))
    
    bars = ax.barh(y_pos, times, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tool_names)
    ax.set_xlabel('Processing Time (milliseconds)', fontweight='bold')
    ax.set_title('Document Processing Speed Comparison', fontweight='bold', fontsize=14)
    ax.invert_yaxis()  # Fastest at top
    
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        width = bar.get_width()
        ax.text(width + max_time * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{time:.0f} ms', ha='left', va='center', fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'processing_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def plot_text_completeness(metrics_list: List[Any], output_path: str) -> str:
    """
    Create bar chart of text completeness scores
    
    Args:
        metrics_list: List of QualityMetrics objects
        output_path: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    setup_plot_style()
    
    # Extract and sort data
    data = [(m.tool_name, m.text_completeness_score) for m in metrics_list]
    data.sort(key=lambda x: x[1], reverse=True)
    
    tool_names = [d[0] for d in data]
    scores = [d[1] for d in data]
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(tool_names)))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(tool_names))
    
    bars = ax.bar(x_pos, scores, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tool_names, rotation=45, ha='right')
    ax.set_ylabel('Completeness Score (%)', fontweight='bold')
    ax.set_xlabel('Tool', fontweight='bold')
    ax.set_title('Text Extraction Completeness Comparison', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add reference line at 100%
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='100% (Best)')
    ax.legend()
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'text_completeness.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def plot_structure_scores(metrics_list: List[Any], output_path: str) -> str:
    """
    Create grouped bar chart showing structure detection components
    
    Args:
        metrics_list: List of QualityMetrics objects
        output_path: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    setup_plot_style()
    
    # Extract data
    tool_names = [m.tool_name for m in metrics_list]
    headings = [m.num_headings for m in metrics_list]
    tables = [m.num_tables for m in metrics_list]
    lists = [m.num_lists for m in metrics_list]
    images = [m.num_images for m in metrics_list]
    
    # Set up bar positions
    x = np.arange(len(tool_names))
    width = 0.2
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create bars
    bars1 = ax.bar(x - 1.5*width, headings, width, label='Headings', color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*width, tables, width, label='Tables', color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*width, lists, width, label='Lists', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*width, images, width, label='Images', color='#f39c12', edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Tool', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Structure Detection Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tool_names, rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'structure_scores.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def plot_similarity_heatmap(results: List[Any], similarity_matrix: np.ndarray, output_path: str) -> str:
    """
    Create heatmap showing pairwise text similarity between tools
    
    Args:
        results: List of IngestionResult objects
        similarity_matrix: NxN numpy array of similarity scores
        output_path: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    setup_plot_style()
    
    # Get tool names for successful results
    valid_results = [r for r in results if r.success and r.text.strip()]
    tool_names = [r.tool_name for r in valid_results]
    
    if len(tool_names) < 2:
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Not enough valid results for similarity analysis',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        output_file = Path(output_path) / 'similarity_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_file)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tool_names)))
    ax.set_yticks(np.arange(len(tool_names)))
    ax.set_xticklabels(tool_names, rotation=45, ha='right')
    ax.set_yticklabels(tool_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(tool_names)):
        for j in range(len(tool_names)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black' if similarity_matrix[i, j] < 0.5 else 'white',
                          fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_title('Text Similarity Between Tools (TF-IDF Cosine)', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'similarity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def plot_overall_scores(metrics_list: List[Any], output_path: str) -> str:
    """
    Create radar/spider chart showing overall quality scores
    
    Args:
        metrics_list: List of QualityMetrics objects
        output_path: Directory to save the plot
        
    Returns:
        Path to saved plot file
    """
    setup_plot_style()
    
    # Extract data
    tool_names = [m.tool_name for m in metrics_list]
    completeness = [m.text_completeness_score for m in metrics_list]
    structure = [m.structure_score for m in metrics_list]
    metadata = [m.metadata_score for m in metrics_list]
    
    # Create grouped bar chart for overall scores
    x = np.arange(len(tool_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width, completeness, width, label='Completeness', 
                   color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, structure, width, label='Structure',
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, metadata, width, label='Metadata',
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Tool', fontweight='bold')
    ax.set_ylabel('Score (0-100)', fontweight='bold')
    ax.set_title('Overall Quality Scores Comparison', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tool_names, rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 105)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path) / 'overall_scores.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def generate_all_visualizations(results: List[Any], metrics_list: List[Any], 
                                similarity_matrix: np.ndarray, output_dir: str) -> Dict[str, str]:
    """
    Generate all visualization charts
    
    Args:
        results: List of IngestionResult objects
        metrics_list: List of QualityMetrics objects
        similarity_matrix: NxN numpy array of similarity scores
        output_dir: Directory to save plots
        
    Returns:
        Dictionary mapping chart names to file paths
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    chart_paths = {}
    
    print("Generating visualizations...")
    
    # Generate each chart
    chart_paths['processing_time'] = plot_processing_time(metrics_list, results, output_dir)
    print("  ✓ Processing time chart created")
    
    chart_paths['text_completeness'] = plot_text_completeness(metrics_list, output_dir)
    print("  ✓ Text completeness chart created")
    
    chart_paths['structure_scores'] = plot_structure_scores(metrics_list, output_dir)
    print("  ✓ Structure detection chart created")
    
    chart_paths['similarity_heatmap'] = plot_similarity_heatmap(results, similarity_matrix, output_dir)
    print("  ✓ Similarity heatmap created")
    
    chart_paths['overall_scores'] = plot_overall_scores(metrics_list, output_dir)
    print("  ✓ Overall scores chart created")
    
    return chart_paths


if __name__ == "__main__":
    print("Visualization Module")
    print("This module is meant to be imported and used by other scripts.")
    print("\nAvailable functions:")
    print("  - generate_all_visualizations(results, metrics_list, similarity_matrix, output_dir)")
    print("  - plot_processing_time(...)")
    print("  - plot_text_completeness(...)")
    print("  - plot_structure_scores(...)")
    print("  - plot_similarity_heatmap(...)")
    print("  - plot_overall_scores(...)")
