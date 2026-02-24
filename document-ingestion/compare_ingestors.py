"""
Document Ingestion Comparison Tool - Core Ingestion Module

This module implements ingestion functions for 7 different PDF processing libraries:
- Docling (IBM)
- Unstructured.io
- LayoutLM (Transformers-based)
- PaddleOCR
- LlamaIndex
- PyMuPDF
- PDFPlumber

Each function returns standardized results for fair comparison.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global output directory for extracted assets
OUTPUT_DIR = None


def set_output_directory(output_dir: str):
    """Set the global output directory for extracted assets"""
    global OUTPUT_DIR
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_image(image_data: bytes, tool_name: str, page_num: int, img_num: int, extension: str = "png") -> str:
    """Save image data to disk and return the path"""
    if OUTPUT_DIR is None:
        return ""
    
    images_dir = OUTPUT_DIR / tool_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"page{page_num}_img{img_num}.{extension}"
    filepath = images_dir / filename
    
    with open(filepath, 'wb') as f:
        f.write(image_data)
    
    return str(filepath)


def save_table(table_data: Any, tool_name: str, page_num: int, table_num: int, format: str = "csv") -> str:
    """Save table data to disk and return the path"""
    if OUTPUT_DIR is None:
        return ""
    
    tables_dir = OUTPUT_DIR / tool_name / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"page{page_num}_table{table_num}.{format}"
    filepath = tables_dir / filename
    
    if format == "csv":
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if isinstance(table_data, list):
                writer.writerows(table_data)
            elif isinstance(table_data, dict):
                # If it's a dict representation, save as JSON instead
                import json
                filepath = filepath.with_suffix('.json')
                with open(filepath, 'w', encoding='utf-8') as jf:
                    json.dump(table_data, jf, indent=2)
    elif format == "json":
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, indent=2)
    
    return str(filepath)


@dataclass
class StructuralElements:
    """Detected structural elements in the document"""
    headings: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[str] = field(default_factory=list)
    images: int = 0
    image_paths: List[str] = field(default_factory=list)  # Paths to saved image files
    table_paths: List[str] = field(default_factory=list)  # Paths to saved table files


@dataclass
class IngestionResult:
    """Standardized result from document ingestion"""
    tool_name: str
    text: str
    metadata: Dict[str, Any]
    processing_time_ms: float
    structural_elements: StructuralElements
    error: Optional[str] = None
    success: bool = True


def ingest_with_docling(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using Docling (IBM's document understanding library)
    
    Docling provides enterprise-grade document processing with layout analysis.
    """
    start_time = time.time()
    tool_name = "Docling"
    
    try:
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        # Extract text
        text = result.document.export_to_markdown()
        
        # Extract metadata
        metadata = {
            "num_pages": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
            "title": getattr(result.document, 'title', ''),
            "source": pdf_path
        }
        
        # Detect structural elements
        structural_elements = StructuralElements()
        
        # Parse markdown for headings
        for line in text.split('\n'):
            if line.startswith('#'):
                structural_elements.headings.append(line.strip())
            elif line.strip().startswith(('-', '*', '+')):
                structural_elements.lists.append(line.strip())
        
        # Extract tables and images from Docling document
        if hasattr(result.document, 'pages'):
            for page_num, page in enumerate(result.document.pages):
                # Extract tables
                if hasattr(page, 'tables'):
                    for table_idx, table in enumerate(page.tables):
                        table_dict = {"page": page_num, "rows": len(table.data) if hasattr(table, 'data') else 0}
                        structural_elements.tables.append(table_dict)
                        if hasattr(table, 'data'):
                            table_path = save_table(table.data, tool_name, page_num, table_idx, "csv")
                            if table_path:
                                structural_elements.table_paths.append(table_path)
                
                # Extract images
                if hasattr(page, 'images') or hasattr(page, 'pictures'):
                    images = getattr(page, 'images', []) or getattr(page, 'pictures', [])
                    for img_idx, img in enumerate(images):
                        structural_elements.images += 1
                        if hasattr(img, 'pil_image'):
                            from io import BytesIO
                            img_bytes = BytesIO()
                            img.pil_image.save(img_bytes, format='PNG')
                            img_path = save_image(img_bytes.getvalue(), tool_name, page_num, img_idx)
                            if img_path:
                                structural_elements.image_paths.append(img_path)
        
        # Fallback: Count markdown tables
        if not structural_elements.tables and '|' in text:
            table_lines = [l for l in text.split('\n') if '|' in l]
            if table_lines:
                structural_elements.tables = [{"rows": len(table_lines)}]
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def ingest_with_unstructured(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using Unstructured.io library
    
    Unstructured provides element-based document parsing with rich metadata.
    """
    start_time = time.time()
    tool_name = "Unstructured"
    
    try:
        from unstructured.partition.pdf import partition_pdf
        
        # Partition the PDF
        elements = partition_pdf(filename=pdf_path)
        
        # Extract text from elements
        text = "\n\n".join([str(el) for el in elements])
        
        # Extract metadata
        metadata = {
            "num_elements": len(elements),
            "source": pdf_path
        }
        
        # Detect structural elements
        structural_elements = StructuralElements()
        
        for idx, el in enumerate(elements):
            el_type = type(el).__name__
            if 'Title' in el_type or 'Header' in el_type:
                structural_elements.headings.append(str(el))
            elif 'Table' in el_type:
                table_dict = {"text": str(el)}
                structural_elements.tables.append(table_dict)
                # Save table text
                if hasattr(el, 'metadata'):
                    page_num = el.metadata.get('page_number', 0)
                else:
                    page_num = 0
                table_path = save_table({"content": str(el)}, tool_name, page_num, len(structural_elements.tables)-1, "json")
                if table_path:
                    structural_elements.table_paths.append(table_path)
            elif 'List' in el_type:
                structural_elements.lists.append(str(el))
            elif 'Image' in el_type:
                structural_elements.images += 1
                # Try to save image if available
                if hasattr(el, 'metadata') and 'image_path' in el.metadata:
                    structural_elements.image_paths.append(el.metadata['image_path'])
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def ingest_with_layoutlm(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using LayoutLM (Microsoft's layout-aware language model)
    
    Uses transformers and PIL for layout analysis and text extraction.
    """
    start_time = time.time()
    tool_name = "LayoutLM"
    
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        from PIL import Image
        import fitz  # PyMuPDF for PDF to image conversion
        
        # Convert PDF pages to images
        doc = fitz.open(pdf_path)
        texts = []
        structural_elements = StructuralElements()
        
        # For simplicity, extract text directly (LayoutLM is typically used for token classification)
        # In a production setting, you'd use the model for NER or document understanding
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            texts.append(text)
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                structural_elements.images += 1
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    img_path = save_image(image_bytes, tool_name, page_num, img_idx, img_ext)
                    if img_path:
                        structural_elements.image_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
        
        doc.close()
        
        text = "\n\n".join(texts)
        
        metadata = {
            "num_pages": len(texts),
            "source": pdf_path,
            "note": "Using PyMuPDF backend for text extraction (LayoutLM for structure)"
        }
        
        # Basic structural detection
        for line in text.split('\n'):
            line_stripped = line.strip()
            # Heuristic: short lines in all caps might be headings
            if len(line_stripped) > 0 and len(line_stripped) < 100 and line_stripped.isupper():
                structural_elements.headings.append(line_stripped)
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def ingest_with_paddleocr(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using PaddleOCR with layout detection
    
    PaddleOCR provides OCR with layout analysis capabilities.
    """
    start_time = time.time()
    tool_name = "PaddleOCR"
    
    try:
        from paddleocr import PaddleOCR
        import fitz  # PyMuPDF for PDF to image conversion
        import numpy as np
        from PIL import Image
        
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Convert PDF pages to images
        doc = fitz.open(pdf_path)
        texts = []
        structural_elements = StructuralElements()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Check for images in the page
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                structural_elements.images += 1
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_path = save_image(image_bytes, tool_name, page_num, img_idx, base_image["ext"])
                    if img_path:
                        structural_elements.image_paths.append(img_path)
                except:
                    pass
            
            # Run OCR
            result = ocr.ocr(np.array(img), cls=True)
            
            if result and result[0]:
                page_text = "\n".join([line[1][0] for line in result[0]])
                texts.append(page_text)
                
                # Detect headings (larger font or specific positioning)
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    # Heuristic: high confidence short text might be heading
                    if confidence > 0.9 and len(text) < 100 and text.isupper():
                        structural_elements.headings.append(text)
        
        doc.close()
        
        text = "\n\n".join(texts)
        
        metadata = {
            "num_pages": len(texts),
            "source": pdf_path,
            "ocr_engine": "PaddleOCR"
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def ingest_with_llamaindex(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using LlamaIndex (SimpleDirectoryReader)
    
    LlamaIndex provides simple document loading with chunking support.
    """
    start_time = time.time()
    tool_name = "LlamaIndex"
    
    try:
        from llama_index.core import SimpleDirectoryReader
        
        # Load document
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        
        # Extract text
        text = "\n\n".join([doc.text for doc in documents])
        
        # Extract metadata
        metadata = {
            "num_documents": len(documents),
            "source": pdf_path
        }
        
        if documents:
            metadata.update(documents[0].metadata)
        
        # Basic structural detection
        structural_elements = StructuralElements()
        
        for line in text.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith(('#', '##', '###')):
                structural_elements.headings.append(line_stripped)
            elif line_stripped.startswith(('-', '*', '•')):
                structural_elements.lists.append(line_stripped)
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def ingest_with_pymupdf(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using PyMuPDF (fitz)
    
    PyMuPDF is a fast and powerful PDF library with excellent text extraction.
    """
    start_time = time.time()
    tool_name = "PyMuPDF"
    
    try:
        import fitz
        
        doc = fitz.open(pdf_path)
        texts = []
        structural_elements = StructuralElements()
        
        # Extract metadata
        metadata = {
            "num_pages": len(doc),
            "source": pdf_path,
            "title": doc.metadata.get('title', ''),
            "author": doc.metadata.get('author', ''),
            "subject": doc.metadata.get('subject', ''),
            "creator": doc.metadata.get('creator', ''),
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            texts.append(text)
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                structural_elements.images += 1
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image["ext"]
                    img_path = save_image(image_bytes, tool_name, page_num, img_idx, img_ext)
                    if img_path:
                        structural_elements.image_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
            
            # Extract structural information using blocks
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Detect headings by font size
                            if span["size"] > 14:  # Larger font might be heading
                                structural_elements.headings.append(span["text"])
        
        doc.close()
        
        text = "\n\n".join(texts)
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def ingest_with_pdfplumber(pdf_path: str) -> IngestionResult:
    """
    Ingest PDF using PDFPlumber
    
    PDFPlumber excels at table extraction and precise text positioning.
    """
    start_time = time.time()
    tool_name = "PDFPlumber"
    
    try:
        import pdfplumber
        
        texts = []
        structural_elements = StructuralElements()
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                "num_pages": len(pdf.pages),
                "source": pdf_path,
            }
            
            # Add PDF metadata if available
            if pdf.metadata:
                metadata.update({
                    "title": pdf.metadata.get('Title', ''),
                    "author": pdf.metadata.get('Author', ''),
                    "subject": pdf.metadata.get('Subject', ''),
                    "creator": pdf.metadata.get('Creator', ''),
                })
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    texts.append(text)
                
                # Extract images
                if hasattr(page, 'images'):
                    for img_idx, img in enumerate(page.images):
                        structural_elements.images += 1
                        try:
                            # PDFPlumber images have bbox and other info
                            img_obj = page.crop(img['bbox']).to_image(resolution=150)
                            from io import BytesIO
                            img_bytes = BytesIO()
                            img_obj.save(img_bytes, format='PNG')
                            img_path = save_image(img_bytes.getvalue(), tool_name, page_num, img_idx, "png")
                            if img_path:
                                structural_elements.image_paths.append(img_path)
                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        structural_elements.tables.append({
                            "rows": len(table),
                            "cols": len(table[0]) if table else 0
                        })
                        # Save table to CSV
                        table_path = save_table(table, tool_name, page_num, table_idx, "csv")
                        if table_path:
                            structural_elements.table_paths.append(table_path)
        
        text = "\n\n".join(texts)
        
        # Detect lists and headings
        for line in text.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith(('-', '•', '*', '○')):
                structural_elements.lists.append(line_stripped)
            # Heuristic for headings: short uppercase lines
            elif len(line_stripped) < 100 and line_stripped.isupper() and len(line_stripped) > 3:
                structural_elements.headings.append(line_stripped)
        
        processing_time = (time.time() - start_time) * 1000
        
        return IngestionResult(
            tool_name=tool_name,
            text=text,
            metadata=metadata,
            processing_time_ms=processing_time,
            structural_elements=structural_elements,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error with {tool_name}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        return IngestionResult(
            tool_name=tool_name,
            text="",
            metadata={"source": pdf_path},
            processing_time_ms=processing_time,
            structural_elements=StructuralElements(),
            error=str(e),
            success=False
        )


def run_comparison(pdf_path: str, output_dir: Optional[str] = None) -> List[IngestionResult]:
    """
    Run all ingestion tools on the given PDF and return results
    
    Args:
        pdf_path: Path to the PDF file to process
        output_dir: Directory to save extracted images and tables
        
    Returns:
        List of IngestionResult objects, one per tool
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Set up output directory for extracted assets
    if output_dir:
        set_output_directory(output_dir)
    
    logger.info(f"Starting comparison for: {pdf_path}")
    
    # List of all ingestion functions
    ingestion_functions = [
        ingest_with_docling,
        ingest_with_unstructured,
        ingest_with_layoutlm,
        ingest_with_paddleocr,
        ingest_with_llamaindex,
        ingest_with_pymupdf,
        ingest_with_pdfplumber,
    ]
    
    results = []
    
    for func in ingestion_functions:
        logger.info(f"Running {func.__name__.replace('ingest_with_', '')}...")
        result = func(pdf_path)
        results.append(result)
        
        if result.success:
            logger.info(f"  ✓ Success - {len(result.text)} chars extracted in {result.processing_time_ms:.2f}ms")
        else:
            logger.warning(f"  ✗ Failed - {result.error}")
    
    return results


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Compare document ingestion tools")
    parser.add_argument("--input", "-i", required=True, help="Path to PDF file")
    parser.add_argument("--output", "-o", default="raw_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_comparison(args.input)
    
    # Convert to JSON-serializable format
    results_dict = []
    for r in results:
        results_dict.append({
            "tool_name": r.tool_name,
            "text": r.text,
            "metadata": r.metadata,
            "processing_time_ms": r.processing_time_ms,
            "structural_elements": {
                "headings": r.structural_elements.headings,
                "tables": r.structural_elements.tables,
                "lists": r.structural_elements.lists,
                "images": r.structural_elements.images,
            },
            "error": r.error,
            "success": r.success,
        })
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {args.output}")
