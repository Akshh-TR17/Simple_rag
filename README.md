# Simple PDF Question Answering System Using RAG

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application built in Python that answers user questions based on the contents of a local PDF file (`sample.pdf`).

This project is designed for a college assignment review, keeping the architecture simple, clean, and easy to explain to faculty without complex frameworks or external cloud dependencies.

---

## 🎯 Project Objectives

1. **Extract and preserve context:** Extract text from every page of `sample.pdf` while retaining the original page numbers.
2. **Chunking:** Split long text into manageable chunks (~500 characters with ~100 characters overlap).
3. **Embeddings:** Generate semantic embeddings locally using Ollama (`nomic-embed-text`).
4. **Vector Store in JSON:** Save chunk metadata, text, page numbers, and embeddings into `vector_store.json` and reuse the index when up-to-date.
5. **NumPy Cosine Similarity:** Calculate cosine similarity from scratch using NumPy to retrieve the top 3 most relevant chunks (no vector databases like Chroma or Pinecone).
6. **Local LLM Answering:** Query Ollama's `qwen2.5:0.5b` to generate answers strictly based on the retrieved context, citing source pages.
7. **FastAPI & Swagger UI:** Provide an interactive API documented at `http://127.0.0.1:8000/docs`.

---

## 🛠️ Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python 3.12+** | Core programming language |
| **Astral uv** | Modern, fast Python environment and package manager |
| **FastAPI** | Lightweight web framework for REST API endpoints |
| **Uvicorn** | ASGI server to run the FastAPI application |
| **Swagger UI** | Automatic interactive API testing dashboard |
| **PyMuPDF (`fitz`)** | Fast text extraction from PDF pages preserving page indices |
| **Ollama** | Local AI runtime for embeddings and text generation |
| **nomic-embed-text** | Local embedding model (768-dimensional vectors) |
| **qwen2.5:0.5b** | Lightweight 0.5-billion parameter local language model |
| **NumPy** | Vector mathematics for cosine similarity search |
| **JSON** | Transparent local storage for vector embeddings |

---

## 💡 What is RAG (Retrieval-Augmented Generation)?

Traditional Large Language Models (LLMs) rely only on their pre-trained knowledge. If you ask a question about your specific document, an LLM might either guess or hallucinate.

**RAG** solves this in three simple steps:
1. **Retrieve:** Find the exact parts of your document that are mathematically closest to the user's question using vector embeddings.
2. **Augment:** Inject those relevant parts into the LLM prompt as context.
3. **Generate:** Ask the LLM to write an answer using *only* that injected context.

---

## 📁 Project Structure

```text
Simple_RAG_Assignment/
│
├── main.py              # Complete RAG pipeline and FastAPI application
├── sample.pdf           # Input PDF file (place your assignment PDF here)
├── vector_store.json    # Auto-generated JSON vector index & metadata
├── pyproject.toml       # Project configuration and uv dependencies
├── uv.lock              # Deterministic uv dependency lockfile
├── requirements.txt     # Submission requirements file (exported via uv)
└── README.md            # Project documentation and guide
```

---

## 📋 Prerequisites

Before running the project, make sure you have:
1. **Windows PowerShell** (or terminal of your choice).
2. **Astral uv** installed.
   - Install via PowerShell (if not already installed):
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
3. **Ollama** installed and running on your computer.
   - Download from: [https://ollama.com](https://ollama.com)

---

## 🦙 Ollama Model Setup

Open Windows PowerShell and pull the two required models:

```powershell
# 1. Pull the embedding model (used to convert text to vectors)
ollama pull nomic-embed-text

# 2. Pull the lightweight LLM (used to generate final answers)
ollama pull qwen2.5:0.5b
```

> **Note:** Ensure Ollama is running in the background (or system tray) before starting the Python application.

---

## ⚙️ Environment Setup & Installation Using `uv`

All dependency and environment management is handled using **uv** (no `pip install` commands needed).

### 1. Open the Project Folder
Open Windows PowerShell in the project directory:
```powershell
cd D:\RAG_PIPELINE
```

### 2. Create the Virtual Environment
```powershell
uv venv
```

### 3. Install Dependencies
Install all required libraries defined in `pyproject.toml`:
```powershell
uv sync
```
*(Alternatively, you can install the dependencies explicitly with `uv add pymupdf ollama numpy fastapi uvicorn`)*.

---

## 📄 Adding Your Assignment PDF

1. Place your PDF file in the root folder of this project.
2. Name the file **`sample.pdf`**.

> **Note:** If `sample.pdf` is missing when you ask a question, the API returns a clear HTTP 404 message telling you to place the file in the project folder.

---

## 🚀 Running the Application

Start the FastAPI development server with `uv run`:

```powershell
uv run uvicorn main:app --reload
```

You will see output similar to:
```text
=================================================================
Simple PDF Question Answering System Starting...
[STARTUP] Found 'sample.pdf'. Initializing vector store...
[RAG] Extracting text from 'sample.pdf'...
[RAG] Creating chunks (size=500, overlap=100)...
[RAG] Generating embeddings for 12 chunks using 'nomic-embed-text'...
[RAG] Saving vector store to 'vector_store.json'...
[STARTUP] Vector store is ready.
Swagger UI is available at: http://127.0.0.1:8000/docs
=================================================================
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 🌐 Testing the API Using Swagger UI

1. Open your browser and navigate to:
   **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
2. You will see two endpoints:
   - **`GET /`**: Welcome message and instructions.
   - **`POST /ask`**: The question answering endpoint.

### How to test `POST /ask`:
1. Click on the **`POST /ask`** box to expand it.
2. Click the **"Try it out"** button in the top right.
3. In the Request body editor, enter your question in JSON format:
   ```json
   {
     "question": "What is machine learning?"
   }
   ```
4. Click the blue **"Execute"** button.
5. Scroll down to **"Responses"** to view the result:
   ```json
   {
     "question": "What is machine learning?",
     "answer": "Machine learning is a subset of artificial intelligence focused on building applications that learn from data and improve their accuracy over time without being explicitly programmed.",
     "source_pages": [1, 2]
   }
   ```

---

## 🧪 Sample Questions to Test

Depending on what your `sample.pdf` covers, you can test questions like:
- *"What is the main topic of this document?"*
- *"Explain the primary methodology discussed on the first page."*
- *"What conclusions or results are highlighted in this paper?"*
- *"Who was the first president of Antarctica?"* (Out-of-domain question to test fallback behavior)

**Expected Fallback Response for unknown questions:**
If the question cannot be answered from the PDF text, the model adheres to its strict prompt instructions and replies:
> `"Sorry, I could not find the answer in the provided PDF."`

---

## 🔬 How Each Component of the Pipeline Works

For faculty review and project explanation, here is a step-by-step breakdown of the functions in `main.py`:

### 1. PDF Text Extraction (`extract_text_from_pdf`)
- PyMuPDF opens `sample.pdf` and iterates page-by-page.
- It extracts plain text and preserves 1-based page numbers (`page_num = page_idx + 1`).

### 2. Text Chunking (`chunk_text`)
- Large pages of text are split into smaller chunks of ~500 characters with an overlap of ~100 characters.
- Overlap prevents sentences from being split in the middle of important thoughts.
- Each chunk preserves its original `page_number` so we can cite sources later.

### 3. Embedding Generation (`get_embedding`)
- Each chunk of text is sent to Ollama's `nomic-embed-text` model.
- The model converts the text into a 768-dimensional numerical vector representing its semantic meaning.

### 4. JSON Vector Store (`build_vector_store` & `is_vector_store_valid`)
- The chunks and their embeddings are saved to `vector_store.json`.
- Metadata such as `pdf_filename`, `pdf_file_size`, `pdf_last_modified`, and `embedding_model` are saved alongside the data.
- When the API starts or receives a request, it checks this metadata. If the PDF has not changed, it reuses the existing JSON file instantly without recomputing embeddings.

### 5. Cosine Similarity & Retrieval (`cosine_similarity` & `retrieve_top_chunks`)
- When the user asks a question, the question is also converted into a 768-dimensional vector using `nomic-embed-text`.
- NumPy computes the cosine similarity between the question vector $\vec{q}$ and every chunk vector $\vec{c}$:
  $$\text{Cosine Similarity} = \frac{\vec{q} \cdot \vec{c}}{\|\vec{q}\| \|\vec{c}\|}$$
- All chunks are sorted by similarity score, and the **top 3** chunks are selected.

### 6. Answer Generation (`generate_answer`)
- The text from the top 3 chunks is formatted into a context block with page markers.
- A strict prompt is sent to `qwen2.5:0.5b` with rules:
  1. Use ONLY the provided context.
  2. Do not use outside knowledge.
  3. If information is missing, output: *"Sorry, I could not find the answer in the provided PDF."*
- The response and sorted source page numbers are returned in JSON.

---

## 📦 Submission File (`requirements.txt`)

If your college submission portal requires a traditional `requirements.txt`, it has been generated using Astral uv:
```powershell
uv export --format requirements-txt --no-hashes -o requirements.txt
```
