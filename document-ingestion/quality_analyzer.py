"""
Quality Analyzer for Document Ingestion Comparison

This module provides metrics to evaluate and compare document ingestion quality:
- Text completeness (character/word count vs baseline)
- Structure detection quality (headings, tables, lists)
- Text similarity between tools (using TF-IDF cosine similarity)
- Metadata extraction completeness
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class QualityMetrics:
    """Quality metrics for a single ingestion result"""
    tool_name: str
    text_completeness_score: float  # 0-100, percentage of max extraction
    structure_score: float  # 0-100, based on detected elements
    metadata_score: float  # 0-100, based on extracted metadata fields  
    word_count: int
    char_count: int
    num_headings: int
    num_tables: int
    num_lists: int
    num_images: int
    metadata_fields: int


def calculate_text_completeness(results: List[Any]) -> Dict[str, float]:
    """
    Calculate text completeness score for each tool
    
    Score is percentage relative to the tool with longest extraction (baseline).
    
    Args:
        results: List of IngestionResult objects
        
    Returns:
        Dictionary mapping tool_name to completeness score (0-100)
    """
    scores = {}
    
    # Find maximum text length as baseline
    max_length = max([len(r.text) for r in results if r.success])
    
    if max_length == 0:
        return {r.tool_name: 0.0 for r in results}
    
    for result in results:
        if result.success:
            score = (len(result.text) / max_length) * 100
        else:
            score = 0.0
        scores[result.tool_name] = round(score, 2)
    
    return scores


def calculate_structure_score(result: Any) -> float:
    """
    Calculate structure detection quality score
    
    Score based on number of structural elements detected:
    - Headings: 2 points each (up to 30 points)
    - Tables: 5 points each (up to 30 points)
    - Lists: 1 point each (up to 20 points)
    - Images: 2 points each (up to 20 points)
    
    Args:
        result: IngestionResult object
        
    Returns:
        Structure score (0-100)
    """
    if not result.success:
        return 0.0
    
    se = result.structural_elements
    
    # Calculate component scores with caps
    heading_score = min(len(se.headings) * 2, 30)
    table_score = min(len(se.tables) * 5, 30)
    list_score = min(len(se.lists) * 1, 20)
    image_score = min(se.images * 2, 20)
    
    total_score = heading_score + table_score + list_score + image_score
    
    return round(total_score, 2)


def calculate_metadata_score(result: Any) -> float:
    """
    Calculate metadata extraction completeness score
    
    Score based on number of meaningful metadata fields extracted:
    - Common fields: title, author, subject, creator, num_pages
    - Each field: 20 points (up to 100 points)
    
    Args:
        result: IngestionResult object
        
    Returns:
        Metadata score (0-100)
    """
    if not result.success:
        return 0.0
    
    important_fields = ['title', 'author', 'subject', 'creator', 'num_pages']
    
    extracted_count = 0
    for field in important_fields:
        value = result.metadata.get(field, '')
        # Count as extracted if not empty
        if value and str(value).strip() != '':
            extracted_count += 1
    
    score = (extracted_count / len(important_fields)) * 100
    
    return round(score, 2)


def count_metadata_fields(metadata: Dict[str, Any]) -> int:
    """
    Count number of non-empty metadata fields
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Count of fields with non-empty values
    """
    count = 0
    for key, value in metadata.items():
        if value and str(value).strip() != '':
            count += 1
    return count


def calculate_pairwise_similarity(results: List[Any]) -> np.ndarray:
    """
    Calculate pairwise text similarity between tools using TF-IDF cosine similarity
    
    Args:
        results: List of IngestionResult objects
        
    Returns:
        NxN numpy array of similarity scores (0-1)
    """
    # Get successful results with text
    valid_results = [r for r in results if r.success and r.text.strip()]
    
    if len(valid_results) < 2:
        # Return identity matrix if not enough valid results
        n = len(results)
        return np.eye(n)
    
    texts = [r.text for r in valid_results]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
    except Exception:
        # If TF-IDF fails (e.g., empty texts), return identity matrix
        n = len(valid_results)
        similarity_matrix = np.eye(n)
    
    return similarity_matrix


def analyze_quality(results: List[Any]) -> List[QualityMetrics]:
    """
    Perform comprehensive quality analysis on all ingestion results
    
    Args:
        results: List of IngestionResult objects
        
    Returns:
        List of QualityMetrics objects
    """
    # Calculate completeness scores
    completeness_scores = calculate_text_completeness(results)
    
    metrics_list = []
    
    for result in results:
        # Calculate individual metrics
        structure_score = calculate_structure_score(result)
        metadata_score = calculate_metadata_score(result)
        
        # Count text stats
        if result.success:
            word_count = len(result.text.split())
            char_count = len(result.text)
        else:
            word_count = 0
            char_count = 0
        
        # Create metrics object
        metrics = QualityMetrics(
            tool_name=result.tool_name,
            text_completeness_score=completeness_scores.get(result.tool_name, 0.0),
            structure_score=structure_score,
            metadata_score=metadata_score,
            word_count=word_count,
            char_count=char_count,
            num_headings=len(result.structural_elements.headings),
            num_tables=len(result.structural_elements.tables),
            num_lists=len(result.structural_elements.lists),
            num_images=result.structural_elements.images,
            metadata_fields=count_metadata_fields(result.metadata),
        )
        
        metrics_list.append(metrics)
    
    return metrics_list


def get_summary_statistics(metrics_list: List[QualityMetrics]) -> Dict[str, Any]:
    """
    Calculate summary statistics across all tools
    
    Args:
        metrics_list: List of QualityMetrics objects
        
    Returns:
        Dictionary with summary statistics
    """
    if not metrics_list:
        return {}
    
    summary = {
        "avg_text_completeness": round(np.mean([m.text_completeness_score for m in metrics_list]), 2),
        "avg_structure_score": round(np.mean([m.structure_score for m in metrics_list]), 2),
        "avg_metadata_score": round(np.mean([m.metadata_score for m in metrics_list]), 2),
        "total_tools": len(metrics_list),
        "max_word_count": max([m.word_count for m in metrics_list]),
        "min_word_count": min([m.word_count for m in metrics_list]),
        "best_completeness_tool": max(metrics_list, key=lambda m: m.text_completeness_score).tool_name,
        "best_structure_tool": max(metrics_list, key=lambda m: m.structure_score).tool_name,
        "best_metadata_tool": max(metrics_list, key=lambda m: m.metadata_score).tool_name,
    }
    
    return summary


def create_comparison_table(metrics_list: List[QualityMetrics]) -> str:
    """
    Create an ASCII table comparing all tools
    
    Args:
        metrics_list: List of QualityMetrics objects
        
    Returns:
        Formatted ASCII table string
    """
    # Header
    table = "\n" + "="*130 + "\n"
    table += f"{'Tool':<15} | {'Words':>8} | {'Chars':>8} | {'Complete%':>10} | {'Structure':>10} | {'Metadata':>10} | {'H/T/L/I':>15}\n"
    table += "="*130 + "\n"
    
    # Rows
    for m in metrics_list:
        htli = f"{m.num_headings}/{m.num_tables}/{m.num_lists}/{m.num_images}"
        table += f"{m.tool_name:<15} | {m.word_count:>8} | {m.char_count:>8} | "
        table += f"{m.text_completeness_score:>9.1f}% | {m.structure_score:>9.1f} | "
        table += f"{m.metadata_score:>9.1f} | {htli:>15}\n"
    
    table += "="*130 + "\n"
    
    # Add legend
    table += "H/T/L/I = Headings/Tables/Lists/Images\n"
    
    return table


if __name__ == "__main__":
    # Example usage for testing
    print("Quality Analyzer Module")
    print("This module is meant to be imported and used by other scripts.")
    print("\nAvailable functions:")
    print("  - analyze_quality(results)")
    print("  - calculate_pairwise_similarity(results)")
    print("  - get_summary_statistics(metrics_list)")
    print("  - create_comparison_table(metrics_list)")
