# Backend/main.py
import os
import uuid
import pickle
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz  # PyMuPDF
import numpy as np
import faiss
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# --- Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo-16k")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K = int(os.getenv("TOP_K", 6))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 64))

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"
META_DIR = DATA_DIR / "meta"

for d in (UPLOAD_DIR, INDEX_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    num_chunks: int

class ChatRequest(BaseModel):
    file_id: str
    question: str
    top_k: int = TOP_K

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# --- Utilities
def extract_text_by_page(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text").strip()
        pages.append(text if text else "")
    doc.close()
    return pages

def chunk_pages(pages: List[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    current = ""
    current_start_page = 0
    page_idx = 0
    while page_idx < len(pages):
        page_text = pages[page_idx]
        candidate = (current + "\n\n" + page_text).strip() if current else page_text
        if len(candidate) >= chunk_size:
            start = 0
            while start < len(candidate):
                end = start + chunk_size
                chunk_text = candidate[start:end]
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start_page": current_start_page,
                    "end_page": page_idx
                })
                start += (chunk_size - overlap)
            current = ""
            current_start_page = page_idx + 1
            page_idx += 1
        else:
            current = candidate
            page_idx += 1
    if current:
        chunk_id = str(uuid.uuid4())
        chunks.append({
            "chunk_id": chunk_id,
            "text": current,
            "start_page": current_start_page,
            "end_page": len(pages) - 1
        })
    return chunks

def batch(iterable, n=BATCH_SIZE):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:i + n]

def embed_texts(texts: List[str], model=EMBEDDING_MODEL) -> List[List[float]]:
    vectors = []
    for b in batch(texts, BATCH_SIZE):
        resp = client.embeddings.create(model=model, input=b)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

def build_faiss_index(embeddings: List[List[float]]):
    arr = np.array(embeddings).astype("float32")
    dim = arr.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(arr)
    return index

# --- Persistence helpers
def save_index(index: faiss.IndexFlatL2, file_id: str):
    path = INDEX_DIR / f"{file_id}.index"
    faiss.write_index(index, str(path))
    return path

def load_index(file_id: str):
    path = INDEX_DIR / f"{file_id}.index"
    if not path.exists():
        raise FileNotFoundError("Index not found")
    return faiss.read_index(str(path))

def save_meta(meta: dict, file_id: str):
    path = META_DIR / f"{file_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    return path

def load_meta(file_id: str):
    path = META_DIR / f"{file_id}.pkl"
    if not path.exists():
        raise FileNotFoundError("Meta not found")
    with open(path, "rb") as f:
        return pickle.load(f)
    
@app.get("/")
def root():
    return {"message": "Backend is running. Use /upload_pdf/ or /chat"}

# --- Endpoints
@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    pages = extract_text_by_page(str(saved_path))
    if sum(len(p) for p in pages) == 0:
        raise HTTPException(status_code=400, detail="No extractable text in PDF")

    chunks = chunk_pages(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    index = build_faiss_index(embeddings)
    save_index(index, file_id)

    meta = {
        "file_id": file_id,
        "filename": file.filename,
        "uploaded_path": str(saved_path),
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "chunk_metadata": chunks
    }
    save_meta(meta, file_id)

    return UploadResponse(file_id=file_id, filename=file.filename, num_chunks=len(chunks))

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        meta = load_meta(req.file_id)
        index = load_index(req.file_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File ID not found")

    q_vec_resp = client.embeddings.create(model=EMBEDDING_MODEL, input=req.question)
    q_vec = np.array(q_vec_resp.data[0].embedding).astype("float32").reshape(1, -1)

    D, I = index.search(q_vec, req.top_k)
    D, I = D.flatten().tolist(), I.flatten().tolist()

    hits = []
    for idx, score in zip(I, D):
        if idx < 0 or idx >= len(meta["chunk_metadata"]):
            continue
        cm = meta["chunk_metadata"][idx]
        hits.append({
            "chunk_id": cm["chunk_id"],
            "start_page": cm["start_page"],
            "end_page": cm["end_page"],
            "score": float(score),
            "text_snippet": cm["text"][:500]
        })

    # Build context
    top_texts = []
    total_len = 0
    max_context_chars = 20_000
    for h in hits:
        t = h["text_snippet"]
        if total_len + len(t) > max_context_chars:
            break
        top_texts.append(t)
        total_len += len(t)
    context = "\n\n---\n\n".join(top_texts)

    system_prompt = (
        "You are a helpful assistant that answers questions ONLY using the provided context. "
        "If the answer cannot be found in the context, say 'I don't know'. "
        "Include page references when possible."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer concisely."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.0
    )

    answer_text = response.choices[0].message.content.strip()
    return ChatResponse(answer=answer_text, sources=hits)
