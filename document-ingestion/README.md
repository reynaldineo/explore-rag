# 📄 Document Ingestion Tool Comparison

A comprehensive benchmarking tool for comparing 7 popular open-source PDF processing libraries. This tool evaluates extraction quality, structure preservation, processing speed, and metadata extraction capabilities.

## 🎯 Overview

This tool compares the following PDF processing libraries:

| Tool | Description | Strengths |
|------|-------------|-----------|
| **[Docling](https://github.com/DS4SD/docling)** | IBM's document understanding library | Enterprise-grade, layout analysis |
| **[Unstructured.io](https://unstructured.io/)** | Element-based document parsing | Rich metadata, multiple formats |
| **[LayoutLM](https://huggingface.co/microsoft/layoutlm-base-uncased)** | Microsoft's layout-aware LM | ML-based structure detection |
| **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** | Baidu's OCR engine | Excellent OCR, layout detection |
| **[LlamaIndex](https://www.llamaindex.ai/)** | Data framework for LLM applications | Simple API, chunking support |
| **[PyMuPDF](https://pymupdf.readthedocs.io/)** | Python bindings for MuPDF | Fast, powerful, rich metadata |
| **[PDFPlumber](https://github.com/jsvine/pdfplumber)** | Plumb a PDF for detailed information | Excellent table extraction |

## 📊 What It Measures

### Extraction Quality
- **Text Completeness**: Percentage of text extracted vs. best performer
- **Word/Character Count**: Total content extracted
- **Text Similarity**: Pairwise cosine similarity using TF-IDF

### Structure Preservation
- **Headings Detection**: Identification of document headings
- **Table Extraction**: Recognition and extraction of tables
- **List Detection**: Bullet points and numbered lists
- **Image Recognition**: Detecting embedded images

### Processing Performance
- **Speed**: Processing time in milliseconds
- **Memory**: Resource utilization (future feature)

### Metadata Extraction
- **Document Properties**: Title, author, subject, creator
- **Page Count**: Number of pages detected
- **Custom Fields**: Tool-specific metadata

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd document-ingestion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Download optional models (for LayoutLM and PaddleOCR):**
   
   These will be downloaded automatically on first use, but you can pre-download:
   
   ```bash
   # PaddleOCR models (optional - auto-downloads)
   python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
   
   # LayoutLM models (optional - auto-downloads via transformers)
   python -c "from transformers import LayoutLMv3Processor; LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base')"
   ```

### Basic Usage

Run comparison on a PDF file:

```bash
python run_comparison.py --input ../docs/sample.pdf
```

This will create a `comparison_output/` directory containing:
- `report.html` - Interactive HTML report
- `comparison_results.json` - Raw results and metrics
- `*.png` - Visualization charts

### Advanced Usage

Specify custom output directory:

```bash
python run_comparison.py --input path/to/document.pdf --output-dir my_results/
```

Run individual components:

```bash
# Just run ingestion (no visualizations)
python compare_ingestors.py --input ../docs/sample.pdf --output raw_results.json

# Test a specific module
python quality_analyzer.py
python visualize.py
```

## 📁 Project Structure

```
document-ingestion/
├── run_comparison.py        # Main orchestrator script
├── compare_ingestors.py     # Core ingestion implementations
├── quality_analyzer.py      # Metrics calculation
├── visualize.py             # Chart generation
├── html_report.py           # HTML report generator
├── README.md                # This file
└── comparison_output/       # Results (created after first run)
    ├── report.html
    ├── comparison_results.json
    ├── processing_time.png
    ├── text_completeness.png
    ├── structure_scores.png
    ├── similarity_heatmap.png
    └── overall_scores.png
```

## 📈 Understanding the Output

### HTML Report

Open `report.html` in your browser to see:

1. **Summary Table**: Overview of all metrics
   - Processing time (ms)
   - Word count
   - Completeness score (%)
   - Structure score (0-100)
   - Metadata score (0-100)
   - Element counts (Headings/Tables/Lists/Images)

2. **Visual Charts**:
   - Processing speed comparison
   - Text extraction completeness
   - Structure detection breakdown
   - Tool similarity heatmap
   - Overall quality scores

3. **Text Previews**: Side-by-side text samples (expandable)

4. **Structure Details**: Detected elements per tool

5. **Metadata**: Extracted document properties

### JSON Results

`comparison_results.json` contains:
```json
{
  "timestamp": "2026-02-07T...",
  "summary": {
    "avg_text_completeness": 85.5,
    "best_completeness_tool": "PyMuPDF",
    "best_structure_tool": "Unstructured",
    ...
  },
  "results": [...],  // Raw extraction results
  "metrics": [...]   // Calculated metrics
}
```

## 🎯 Scoring System

### Text Completeness Score (0-100%)
- Calculated relative to the tool with longest extraction
- 100% = extracted the most text
- Lower scores indicate incomplete extraction

### Structure Score (0-100)
Weighted scoring:
- **Headings**: 2 points each (max 30)
- **Tables**: 5 points each (max 30)
- **Lists**: 1 point each (max 20)
- **Images**: 2 points each (max 20)

### Metadata Score (0-100%)
Based on extraction of key fields:
- Title, Author, Subject, Creator, Page Count
- 20 points per field = 100% max

## 🔧 Troubleshooting

### Common Issues

**1. Import errors for specific libraries**

If a library fails to import, the tool will continue with others. Check:

```bash
pip install docling unstructured paddleocr llama-index pymupdf pdfplumber transformers
```

**2. PaddleOCR model download fails**

PaddleOCR downloads models on first use. If it fails:
- Check internet connection
- Manually download from [PaddleOCR models](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/models_list_en.md)

**3. Memory issues with large PDFs**

For PDFs >100 pages:
- Process one page at a time
- Increase system swap space
- Use a machine with more RAM

**4. LayoutLM runs but doesn't detect structure well**

LayoutLM in this implementation uses PyMuPDF backend for text extraction. For full LayoutLM capabilities:
- Fine-tune the model on your document type
- Use document image understanding APIs

**5. Unstructured.io takes too long**

Unstructured can be slow on complex PDFs. To speed up:
- Use `hi_res` strategy with caution
- Consider using their API service for better performance

### Dependencies Issues

If you encounter missing dependencies:

```bash
# Core dependencies
pip install numpy scikit-learn matplotlib pillow

# Document processing
pip install pymupdf pdfplumber

# OCR and ML
pip install paddleocr transformers torch

# Framework-specific
pip install docling unstructured llama-index

# Web dependencies for Unstructured
pip install "unstructured[pdf]"
```

## 🎓 Interpretation Guide

### Which Tool Should I Use?

**For Speed**:
- ✅ PyMuPDF - Fastest text extraction
- ✅ PDFPlumber - Good balance of speed and features

**For Accuracy**:
- ✅ Unstructured - Best structure detection
- ✅ Docling - Enterprise-grade quality

**For Tables**:
- ✅ PDFPlumber - Excellent table extraction
- ✅ Unstructured - Good table detection

**For OCR (scanned PDFs)**:
- ✅ PaddleOCR - Best OCR capabilities
- ✅ Unstructured - Good OCR integration

**For Simplicity**:
- ✅ LlamaIndex - Simplest API
- ✅ PyMuPDF - Straightforward, well-documented

**For RAG/LLM Applications**:
- ✅ LlamaIndex - Built for LLM workflows
- ✅ Unstructured - Rich element metadata

### Similarity Scores

- **High Similarity (>0.9)**: Tools extract nearly identical text
- **Medium Similarity (0.7-0.9)**: Core content matches, formatting differs
- **Low Similarity (<0.7)**: Significant differences in extraction

Low similarity may indicate:
- One tool is missing content
- Different handling of special characters
- OCR vs native text extraction

## 🔍 Example Commands

```bash
# Basic comparison
python run_comparison.py --input ../docs/climate-change.md

# Custom output location
python run_comparison.py -i report.pdf -o results_2026/

# Test with different PDFs
python run_comparison.py -i ../docs/enterprise-billing.pdf -o billing_results/
python run_comparison.py -i academic_paper.pdf -o paper_analysis/

# Run from parent directory
cd ..
python document-ingestion/run_comparison.py --input docs/sample.pdf
```

## 📚 Additional Resources

### Tool Documentation
- [Docling Docs](https://github.com/DS4SD/docling)
- [Unstructured Docs](https://unstructured-io.github.io/unstructured/)
- [LayoutLM Paper](https://arxiv.org/abs/1912.13318)
- [PaddleOCR Docs](https://github.com/PaddlePaddle/PaddleOCR)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [PyMuPDF Docs](https://pymupdf.readthedocs.io/)
- [PDFPlumber Docs](https://github.com/jsvine/pdfplumber)

### Best Practices
1. **Test multiple PDFs**: Results vary by document type
2. **Check structure detection**: View samples, not just counts
3. **Verify text quality**: Read preview sections carefully
4. **Consider your use case**: Speed vs. accuracy tradeoffs
5. **Combine tools**: Use best tool for each document type

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Additional PDF libraries (e.g., PDFMiner, Camelot)
- More detailed structure analysis
- Character encoding comparison
- Memory usage profiling
- Batch processing support
- CI/CD integration for testing

## 📝 License

This comparison tool is open source. Individual libraries have their own licenses:
- Check each tool's repository for licensing details
- Ensure compliance when using in production

## 🙋 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review individual tool documentation
3. Open an issue with sample PDF and error logs

---

**Note**: This tool is for evaluation purposes. Always test libraries with your specific document types before production use.
