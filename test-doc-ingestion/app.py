"""
Document Ingestion Pipeline
===========================

This pipeline extracts:
- Text
- Tables
- Images
- Layout blocks
- OCR text from images
- Charts/figures (as images)

It uses ONLY free/open-source libraries:
- PyMuPDF (fitz)
- pdfplumber
- unstructured
- paddleocr
- PIL
- matplotlib (for visualization)
- llama-index (for structured document loading)

Install dependencies:

pip install pymupdf pdfplumber unstructured[local-inference] paddleocr pillow matplotlib llama-index

(Optional but recommended):
pip install opencv-python layoutparser torch torchvision

"""

import os
import uuid
import fitz  # PyMuPDF
import pdfplumber
from unstructured.partition.pdf import partition_pdf
from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
from llama_index.core import SimpleDirectoryReader

# -----------------------------
# Config
# -----------------------------
INPUT_PDF = "./../docs/DocLayNet.pdf"
OUTPUT_DIR = "extracted_outputs"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OCR_DIR = os.path.join(OUTPUT_DIR, "ocr")
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# -----------------------------
# 1. TEXT + LAYOUT + TABLES (Unstructured)
# -----------------------------
def extract_with_unstructured(pdf_path):
    print("[+] Extracting with Unstructured...")
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi-res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        extract_image_block_types=["Image", "Table"],
    )

    text_blocks = []
    tables = []

    for el in elements:
        if el.category == "Table":
            tables.append(el)
        else:
            text_blocks.append(el)

    # Save text
    text_file = os.path.join(TEXT_DIR, "unstructured_text.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for block in text_blocks:
            f.write(f"[{block.category}]\n{block.text}\n\n")

    # Save tables
    for i, table in enumerate(tables):
        table_file = os.path.join(TABLE_DIR, f"table_{i+1}.txt")
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(table.text)

    print(f"    → Saved text to {text_file}")
    print(f"    → Extracted {len(tables)} tables")

# -----------------------------
# 2. RAW TEXT + TABLES (pdfplumber)
# -----------------------------
def extract_with_pdfplumber(pdf_path):
    print("[+] Extracting with pdfplumber...")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_file = os.path.join(TEXT_DIR, f"pdfplumber_page_{i+1}.txt")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)

            tables = page.extract_tables()
            for j, table in enumerate(tables):
                table_file = os.path.join(TABLE_DIR, f"pdfplumber_table_p{i+1}_{j+1}.txt")
                with open(table_file, "w", encoding="utf-8") as f:
                    for row in table:
                        f.write("\t".join([str(cell) if cell else "" for cell in row]) + "\n")

# -----------------------------
# 3. IMAGES + FIGURES (PyMuPDF)
# -----------------------------
def extract_images_with_pymupdf(pdf_path):
    print("[+] Extracting images with PyMuPDF...")
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            img_name = f"page_{page_index+1}_img_{img_index+1}.{ext}"
            img_path = os.path.join(IMG_DIR, img_name)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(img_path)

    print(f"    → Extracted {len(image_paths)} images")
    return image_paths

# -----------------------------
# 4. OCR ON EXTRACTED IMAGES (PaddleOCR)
# -----------------------------
def ocr_images(image_paths):
    print("[+] Running OCR on extracted images...")
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang="en")

        for img_path in image_paths:
            result = ocr.ocr(img_path)
            ocr_text_file = os.path.join(OCR_DIR, os.path.basename(img_path) + ".txt")
            with open(ocr_text_file, "w", encoding="utf-8") as f:
                for line in result:
                    for box in line:
                        f.write(box[1][0] + "\n")
    except Exception as e:
        print(f"    → OCR failed: {e}")
        print("    → Skipping OCR")

# -----------------------------
# 5. LLAMAINDEX DOCUMENT LOADING
# -----------------------------
def extract_with_llamaindex(pdf_path):
    print("[+] Extracting with LlamaIndex SimpleDirectoryReader...")
    temp_dir = os.path.join(OUTPUT_DIR, "llama_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Copy PDF
    import shutil
    shutil.copy(pdf_path, temp_dir)

    documents = SimpleDirectoryReader(temp_dir).load_data()
    out_file = os.path.join(TEXT_DIR, "llamaindex_text.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text + "\n\n")

    print(f"    → Saved LlamaIndex extracted text to {out_file}")

# -----------------------------
# 6. VISUALIZATION (SEE WHAT WAS EXTRACTED)
# -----------------------------
def visualize_extracted_images(image_paths, max_images=9):
    print("[+] Visualizing extracted images...")
    images = image_paths[:max_images]
    cols = 3
    rows = (len(images) + cols - 1) // cols

    plt.figure(figsize=(12, 4 * rows))
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(os.path.basename(img_path))

    plt.tight_layout()
    vis_path = os.path.join(VIS_DIR, "extracted_images_preview.png")
    plt.savefig(vis_path)
    plt.show()
    print(f"    → Saved preview to {vis_path}")

# -----------------------------
# 7. MAIN PIPELINE
# -----------------------------
def run_pipeline(pdf_path):
    print("=" * 60)
    print(f"📄 Processing: {pdf_path}")
    print("=" * 60)

    extract_with_unstructured(pdf_path)
    extract_with_pdfplumber(pdf_path)
    image_paths = extract_images_with_pymupdf(pdf_path)
    ocr_images(image_paths)
    extract_with_llamaindex(pdf_path)
    visualize_extracted_images(image_paths)

    print("\n✅ Pipeline completed successfully!")
    print(f"📂 All outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline(INPUT_PDF)
