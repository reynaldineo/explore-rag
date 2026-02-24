import os
import io
import json
import base64
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity

# Optional: CLIP (image + text embeddings)
import torch
import clip

import requests

def call_ollama_chat(model: str, messages: list, images: list = None) -> str:
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    if images and messages and messages[-1]["role"] == "user":
        messages[-1]["images"] = images
    try:
        response = requests.post(url, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result["message"]["content"]
        else:
            logger.error(f"Ollama API error: {response.text}")
            return "Error: Unable to generate response from Ollama."
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return "Error: Ollama not reachable."

# -----------------------
# Configuration
# -----------------------

DATA_DIR = "data_multimodal"
EMBEDDING_DIM = 512
TOP_K = 5

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Models
# -----------------------

@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    embedding: np.ndarray


@dataclass
class ImageChunk:
    image_id: str
    document_id: str
    image_path: str
    caption: str
    ocr_text: str
    embedding: np.ndarray


@dataclass
class RetrievalResult:
    id: str
    type: str  # "text" or "image"
    score: float
    content: str
    reference: Dict


# -----------------------
# Embedding Models
# -----------------------

class CLIPEmbeddingModel:
    def __init__(self, model_name: str = "ViT-B/16", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()

    def embed_image(self, images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
            embeddings = self.model.encode_image(tensors)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()


# -----------------------
# Image Processing
# -----------------------

class ImageProcessor:
    def __init__(self, embedding_model: CLIPEmbeddingModel):
        self.embedding_model = embedding_model

    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[str, Image.Image]]:
        doc = fitz.open(pdf_path)
        images = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"{os.path.basename(pdf_path)}_p{page_index}_img{img_index}"
                images.append((image_id, image))

        return images

    def run_ocr(self, image: Image.Image) -> str:
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def generate_caption(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        messages = [{"role": "user", "content": "Generate a detailed caption for this image."}]
        caption = call_ollama_chat("qwen3-vl:8b", messages, [img_base64])
        return caption

    def encode_image(self, image: Image.Image) -> np.ndarray:
        return self.embedding_model.embed_image([image])[0]


# -----------------------
# Text Processing
# -----------------------

class TextProcessor:
    def __init__(self, embedding_model: CLIPEmbeddingModel):
        self.embedding_model = embedding_model

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + chunk_size, length)
            chunks.append(text[start:end])
            start = end - overlap

        return chunks

    def encode_text(self, texts: List[str]) -> np.ndarray:
        return self.embedding_model.embed_text(texts)


# -----------------------
# Multi-Modal Vector Store
# -----------------------

class MultiModalVectorStore:
    def __init__(self):
        self.text_chunks: List[TextChunk] = []
        self.image_chunks: List[ImageChunk] = []

    def add_text_chunk(self, chunk: TextChunk):
        self.text_chunks.append(chunk)

    def add_image_chunk(self, chunk: ImageChunk):
        self.image_chunks.append(chunk)

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> List[RetrievalResult]:
        results = []

        if self.text_chunks:
            text_embeddings = np.vstack([c.embedding for c in self.text_chunks])
            text_scores = cosine_similarity([query_embedding], text_embeddings)[0]

            for idx, score in enumerate(text_scores):
                chunk = self.text_chunks[idx]
                results.append(RetrievalResult(
                    id=chunk.chunk_id,
                    type="text",
                    score=float(score),
                    content=chunk.text,
                    reference={"document_id": chunk.document_id}
                ))

        if self.image_chunks:
            image_embeddings = np.vstack([c.embedding for c in self.image_chunks])
            image_scores = cosine_similarity([query_embedding], image_embeddings)[0]

            for idx, score in enumerate(image_scores):
                chunk = self.image_chunks[idx]
                results.append(RetrievalResult(
                    id=chunk.image_id,
                    type="image",
                    score=float(score),
                    content=chunk.caption or chunk.ocr_text,
                    reference={
                        "document_id": chunk.document_id,
                        "image_path": chunk.image_path
                    }
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# -----------------------
# Multi-Modal RAG System
# -----------------------

class MultiModalRAG:
    def __init__(self):
        self.embedding_model = CLIPEmbeddingModel()
        self.image_processor = ImageProcessor(self.embedding_model)
        self.text_processor = TextProcessor(self.embedding_model)
        self.vector_store = MultiModalVectorStore()

    # -------- Ingestion --------

    def ingest_document(self, document_id: str, text: Optional[str] = None, pdf_path: Optional[str] = None):
        if text:
            self._ingest_text(document_id, text)

        if pdf_path:
            self._ingest_pdf(document_id, pdf_path)

    def _ingest_text(self, document_id: str, text: str):
        chunks = self.text_processor.split_text(text)
        embeddings = self.text_processor.encode_text(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_text_{i}"
            self.vector_store.add_text_chunk(TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk,
                embedding=emb,
            ))

        logger.info(f"Ingested {len(chunks)} text chunks for document {document_id}")

    def _ingest_pdf(self, document_id: str, pdf_path: str):
        images = self.image_processor.extract_images_from_pdf(pdf_path)

        for image_id, image in images:
            caption = self.image_processor.generate_caption(image)
            ocr_text = self.image_processor.run_ocr(image)
            embedding = self.image_processor.encode_image(image)

            image_path = os.path.join(DATA_DIR, f"{image_id}.png")
            image.save(image_path)

            self.vector_store.add_image_chunk(ImageChunk(
                image_id=image_id,
                document_id=document_id,
                image_path=image_path,
                caption=caption,
                ocr_text=ocr_text,
                embedding=embedding,
            ))

        logger.info(f"Ingested {len(images)} images for document {document_id}")

    # -------- Querying --------

    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        query_embedding = self.embedding_model.embed_text([question])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)

        answer = self._generate_answer(question, results)

        return {
            "question": question,
            "answer": answer,
            "results": [self._format_result(r) for r in results],
        }

    # -------- Answer Generation --------

    def _generate_answer(self, question: str, results: List[RetrievalResult]) -> str:
        text_parts = [r.content for r in results if r.type == "text"]
        text_context = "\n".join(text_parts)
        image_bases = [self._load_image_base64(r.reference["image_path"]) for r in results if r.type == "image"]
        prompt = f"Based on the following text and images, answer the question: {question}\n\nText:\n{text_context}"
        messages = [{"role": "user", "content": prompt}]
        answer = call_ollama_chat("qwen3-vl:8b", messages, image_bases)
        return answer

    # -------- Utilities --------

    def _format_result(self, result: RetrievalResult) -> Dict:
        output = {
            "id": result.id,
            "type": result.type,
            "score": round(result.score, 4),
            "content": result.content,
            "reference": result.reference,
        }

        if result.type == "image":
            output["image_base64"] = self._load_image_base64(result.reference["image_path"])

        return output

    def _load_image_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded


# -----------------------
# Example Usage
# -----------------------

def main():
    rag = MultiModalRAG()

    # Example ingestion
    rag.ingest_document(
        document_id="sample_doc",
        text="This document explains the system architecture and includes diagrams.",
        pdf_path="./../docs/amazon-esg-2024-page-9.pdf",  # Replace with your PDF path
    )

    # Example query
    result = rag.query("Show me the architecture diagram")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
