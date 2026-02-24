"""
11_multimodal_rag.py
Multi-Modal RAG (Text + Images)
"""

import fitz  # PyMuPDF
import os
import base64
import uuid
from PIL import Image
import pytesseract
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

# ---------------------------
# CONFIG
# ---------------------------

DATA_DIR = "data"
IMAGE_DIR = "extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text + image embedding model (CLIP-like)
EMBED_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

# Image captioning model
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# ---------------------------
# LOAD MODELS
# ---------------------------

embedder = SentenceTransformer(EMBED_MODEL_NAME)

caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
caption_model = BlipForConditionalGeneration.from_pretrained(
    CAPTION_MODEL_NAME
).to(DEVICE)

# ---------------------------
# DATA STRUCTURES
# ---------------------------

documents = []   # text chunks
images = []      # image metadata
embeddings = []  # unified embedding list

# ---------------------------
# PDF INGESTION
# ---------------------------

def extract_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        # --- Extract text ---
        text = page.get_text()
        if text.strip():
            documents.append({
                "id": str(uuid.uuid4()),
                "type": "text",
                "content": text,
                "source": pdf_path,
                "page": page_num
            })

        # --- Extract images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            image_id = str(uuid.uuid4())
            image_path = os.path.join(
                IMAGE_DIR, f"{image_id}.{ext}"
            )

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "id": image_id,
                "type": "image",
                "path": image_path,
                "source": pdf_path,
                "page": page_num
            })

# ---------------------------
# OCR
# ---------------------------

def ocr_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# ---------------------------
# IMAGE CAPTIONING
# ---------------------------

def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = caption_processor(
        image, return_tensors="pt"
    ).to(DEVICE)

    out = caption_model.generate(**inputs, max_new_tokens=50)
    caption = caption_processor.decode(
        out[0], skip_special_tokens=True
    )
    return caption

# ---------------------------
# EMBEDDING GENERATION
# ---------------------------

def embed_text(text):
    return embedder.encode(text, normalize_embeddings=True)

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return embedder.encode(image, normalize_embeddings=True)

# ---------------------------
# BUILD INDEX
# ---------------------------

def build_index():
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    for doc in documents:
        vec = embed_text(doc["content"])
        embeddings.append(vec)
        index.add(np.array([vec]))

    for img in images:
        caption = caption_image(img["path"])
        ocr_text = ocr_image(img["path"])

        img["caption"] = caption
        img["ocr"] = ocr_text

        combined_text = f"{caption}\n{ocr_text}"
        vec = embed_text(combined_text)

        embeddings.append(vec)
        index.add(np.array([vec]))

    return index

# ---------------------------
# SEARCH
# ---------------------------

def search(query, index, k=5):
    q_vec = embed_text(query)
    scores, indices = index.search(
        np.array([q_vec]), k
    )

    results = []
    all_items = documents + images

    for idx in indices[0]:
        results.append(all_items[idx])

    return results

# ---------------------------
# IMAGE DISPLAY (BASE64)
# ---------------------------

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------------------
# ANSWER GENERATION (LLM STUB)
# ---------------------------

def generate_answer(query, retrieved_items):
    """
    Replace with:
    - GPT-4V
    - Claude 3 Vision
    - LLaVA
    """
    answer = f"Query: {query}\n\nRelevant content:\n"

    for item in retrieved_items:
        if item["type"] == "text":
            answer += f"- Text (page {item['page']}): {item['content'][:200]}...\n"
        else:
            answer += f"- Image (page {item['page']}): {item['caption']}\n"

    return answer

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"
    extract_pdf_content(pdf_path)

    index = build_index()

    query = "show me the architecture diagram"
    results = search(query, index)

    answer = generate_answer(query, results)
    print(answer)

    # Display images
    for r in results:
        if r["type"] == "image":
            print(f"\nImage (base64 preview): {image_to_base64(r['path'])[:200]}...")
