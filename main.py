"""
Simple PDF Question Answering System Using RAG
===============================================
A simple, beginner-friendly Retrieval-Augmented Generation (RAG) application.
Built with:
  - FastAPI: REST API & Swagger UI documentation
  - PyMuPDF: Extracting text from PDF pages
  - Ollama: Local AI inference (nomic-embed-text & qwen2.5:0.5b)
  - NumPy: Cosine similarity search
  - JSON: Lightweight local vector store
"""

import os
import json
from contextlib import asynccontextmanager
import pymupdf
import numpy as np
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
PDF_FILE_PATH = "sample.pdf"
VECTOR_STORE_PATH = "vector_store.json"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:0.5b"
CHUNK_SIZE = 500       # Approximate chunk size in characters
CHUNK_OVERLAP = 100    # Overlap between consecutive chunks in characters
TOP_K = 3              # Number of relevant chunks to retrieve for context
FALLBACK_ANSWER = "Sorry, I could not find the answer in the provided PDF."

# In-memory cache for loaded chunks
_cached_chunks = None
_cached_mtime = None


# ============================================================================
# 1. PDF TEXT EXTRACTION
# ============================================================================
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from every page of the given PDF file using PyMuPDF.
    Preserves the original 1-indexed page numbers for source citations.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"The file '{pdf_path}' was not found. Please place your PDF in the project folder."
        )

    doc = pymupdf.open(pdf_path)
    pages_data = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text().strip()
        # Only keep pages that contain readable text
        if text:
            pages_data.append({
                "page_number": page_idx + 1,
                "text": text
            })

    doc.close()
    return pages_data


# ============================================================================
# 2. TEXT CHUNKING
# ============================================================================
def chunk_text(pages_data: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits the extracted page text into smaller overlapping chunks.
    Attaches the original page number to each chunk so we can cite sources.
    """
    chunks = []
    chunk_id = 1
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    for page in pages_data:
        text = page["text"]
        page_num = page["page_number"]

        # If page text is shorter than chunk_size, keep it as a single chunk
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "page_number": page_num
            })
            chunk_id += 1
            continue

        # Split long text with sliding window
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "page_number": page_num
                })
                chunk_id += 1
            start += step

    return chunks


# ============================================================================
# 3. EMBEDDING GENERATION
# ============================================================================
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Generates a dense vector embedding for a string using Ollama's nomic-embed-text.
    """
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate embedding with Ollama model '{model}'. "
            f"Please ensure Ollama is running and run 'ollama pull {model}'. Error: {e}"
        )


# ============================================================================
# 4. JSON VECTOR STORE
# ============================================================================
def is_vector_store_valid(pdf_path: str, store_path: str, embedding_model: str) -> bool:
    """
    Checks if an existing vector_store.json is up-to-date with the current PDF.
    Validates PDF filename, file size, last modified time, and embedding model.
    """
    if not os.path.exists(store_path) or not os.path.exists(pdf_path):
        return False

    try:
        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        chunks = data.get("chunks", [])

        if not chunks:
            return False

        # Compare file metadata
        current_size = os.path.getsize(pdf_path)
        current_mtime = os.path.getmtime(pdf_path)

        matches_file = (
            meta.get("pdf_filename") == os.path.basename(pdf_path) and
            meta.get("pdf_file_size") == current_size and
            abs(meta.get("pdf_last_modified", 0) - current_mtime) < 1e-3
        )
        matches_model = (meta.get("embedding_model") == embedding_model)

        return matches_file and matches_model
    except Exception:
        return False


def build_vector_store(pdf_path: str, store_path: str, embedding_model: str) -> list[dict]:
    """
    Extracts text, creates chunks, computes embeddings, and saves to vector_store.json.
    """
    print(f"[RAG] Extracting text from '{pdf_path}'...")
    pages_data = extract_text_from_pdf(pdf_path)
    if not pages_data:
        raise ValueError(f"The PDF file '{pdf_path}' contains no extractable text.")

    print(f"[RAG] Creating chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_text(pages_data, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("Could not create any text chunks from the PDF.")

    print(f"[RAG] Generating embeddings for {len(chunks)} chunks using '{embedding_model}'...")
    for idx, chunk in enumerate(chunks, 1):
        chunk["embedding"] = get_embedding(chunk["text"], model=embedding_model)

    store_data = {
        "metadata": {
            "pdf_filename": os.path.basename(pdf_path),
            "pdf_file_size": os.path.getsize(pdf_path),
            "pdf_last_modified": os.path.getmtime(pdf_path),
            "embedding_model": embedding_model,
            "total_chunks": len(chunks)
        },
        "chunks": chunks
    }

    print(f"[RAG] Saving vector store to '{store_path}'...")
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store_data, f, indent=2)

    return chunks


def load_or_build_vector_store() -> list[dict]:
    """
    Loads existing vector_store.json if valid, or rebuilds it from sample.pdf.
    """
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(
            f"'{PDF_FILE_PATH}' was not found. Please place your assignment PDF named '{PDF_FILE_PATH}' in the project folder."
        )

    if is_vector_store_valid(PDF_FILE_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL):
        print(f"[RAG] Reusing valid cached index from '{VECTOR_STORE_PATH}'.")
        with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunks", [])
    else:
        print(f"[RAG] Vector store missing or out of date. Building new index...")
        return build_vector_store(PDF_FILE_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL)


def get_chunks() -> list[dict]:
    """
    Returns chunks from memory cache if up to date, otherwise loads/rebuilds them.
    """
    global _cached_chunks, _cached_mtime
    if not os.path.exists(PDF_FILE_PATH):
        return []

    current_mtime = os.path.getmtime(PDF_FILE_PATH)
    if _cached_chunks is not None and _cached_mtime == current_mtime:
        return _cached_chunks

    _cached_chunks = load_or_build_vector_store()
    _cached_mtime = current_mtime
    return _cached_chunks


# ============================================================================
# 5. SIMILARITY SEARCH (COSINE SIMILARITY)
# ============================================================================
def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calculates cosine similarity between two vectors using NumPy.
    Formula: dot(a, b) / (norm(a) * norm(b))
    """
    a = np.array(vector_a, dtype=np.float32)
    b = np.array(vector_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def retrieve_top_chunks(query: str, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Converts question to an embedding and retrieves the top K most similar chunks.
    """
    query_embedding = get_embedding(query, model=EMBEDDING_MODEL)

    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((score, chunk))

    # Sort descending by similarity score
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [chunk for _, chunk in scored_chunks[:top_k]]


# ============================================================================
# 6. ANSWER GENERATION
# ============================================================================
def generate_answer(question: str, top_chunks: list[dict]) -> str:
    """
    Generates an answer using Ollama's qwen2.5:0.5b model based solely on retrieved context.
    """
    # Combine retrieved chunks into context text
    context_sections = []
    for chunk in top_chunks:
        context_sections.append(f"[Page {chunk['page_number']}]: {chunk['text']}")

    context_str = "\n\n".join(context_sections)

    prompt = f"""You are a helpful assistant answering questions strictly based on the provided PDF context.

Rules:
1. Answer the question using ONLY the context provided below.
2. Do NOT use outside knowledge or make up facts.
3. If the context does not contain enough information to answer the question, reply EXACTLY with:
"{FALLBACK_ANSWER}"
4. Keep your answer clear, accurate, and concise.

Context:
{context_str}

Question:
{question}

Answer:"""

    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return response["response"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate answer with Ollama model '{LLM_MODEL}'. "
            f"Please ensure Ollama is running and run 'ollama pull {LLM_MODEL}'. Error: {e}"
        )


# ============================================================================
# 7. FASTAPI APPLICATION & LIFESPAN
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan startup and shutdown handler.
    Checks PDF availability and pre-loads the vector store if sample.pdf is present.
    """
    print("=" * 65)
    print("Simple PDF Question Answering System Starting...")
    if os.path.exists(PDF_FILE_PATH):
        print(f"[STARTUP] Found '{PDF_FILE_PATH}'. Initializing vector store...")
        try:
            get_chunks()
            print("[STARTUP] Vector store is ready.")
        except Exception as err:
            print(f"[STARTUP WARNING] Could not initialize index: {err}")
    else:
        print(f"[STARTUP NOTE] '{PDF_FILE_PATH}' was not found in the project folder.")
        print(f"Please place your assignment PDF named '{PDF_FILE_PATH}' here to query it.")
    print("Swagger UI is available at: http://127.0.0.1:8000/docs")
    print("=" * 65)
    yield
    print("[SHUTDOWN] API server stopped.")


app = FastAPI(
    title="Simple PDF Question Answering System Using RAG",
    description="A beginner-friendly PDF Question Answering API using PyMuPDF, Ollama, and Cosine Similarity.",
    version="1.0.0",
    lifespan=lifespan
)


# Request and Response schemas
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_pages: list[int]


@app.get("/")
def read_root():
    """
    Welcome endpoint providing basic API info and usage instructions.
    """
    return {
        "message": "Welcome to the Simple PDF Question Answering System API!",
        "status": "Running",
        "instructions": "Place your 'sample.pdf' inside the project folder, then use POST /ask or Swagger UI at /docs.",
        "endpoints": {
            "GET /": "Welcome message and API instructions",
            "POST /ask": "Submit a question about sample.pdf and receive an answer with source page numbers",
            "Swagger UI": "http://127.0.0.1:8000/docs"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Accepts a question, retrieves the top 3 relevant chunks from sample.pdf,
    and returns an answer generated by qwen2.5:0.5b along with source page numbers.
    """
    cleaned_question = request.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty. Please provide a valid question."
        )

    # Check if sample.pdf is available
    if not os.path.exists(PDF_FILE_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"'{PDF_FILE_PATH}' was not found. Please place your assignment PDF named '{PDF_FILE_PATH}' inside the project folder."
        )

    # Load or refresh vector store
    try:
        chunks = get_chunks()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF into vector store: {str(e)}"
        )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"The file '{PDF_FILE_PATH}' contains no extractable text chunks."
        )

    # Perform similarity search to get top 3 chunks
    try:
        top_chunks = retrieve_top_chunks(cleaned_question, chunks, top_k=TOP_K)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during similarity search: {str(e)}"
        )

    # Generate answer with Ollama
    try:
        answer = generate_answer(cleaned_question, top_chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

    # Only cite source pages if the answer was found in the PDF.
    # When the fallback answer is returned, source_pages must be an empty list.
    if answer.strip() == FALLBACK_ANSWER or FALLBACK_ANSWER.lower() in answer.lower():
        source_pages = []
    else:
        # Extract distinct source page numbers sorted in ascending order
        source_pages = sorted(list(set(chunk["page_number"] for chunk in top_chunks)))

    return AnswerResponse(
        question=cleaned_question,
        answer=answer,
        source_pages=source_pages
    )
