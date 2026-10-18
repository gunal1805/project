"""
RAG CORE - Document Q&A with Gemini
------------------------------------
The whole pipeline in one file:
  1. Load    -> read text from each PDF page
  2. Chunk   -> split pages into small overlapping pieces
  3. Embed   -> turn each chunk into a vector (Gemini embeddings)
  4. Retrieve-> find the chunks closest in meaning to the question
  5. Generate-> ask Gemini to answer using ONLY those chunks

Run it directly for a terminal/Colab version:
    python rag_core.py your_document.pdf
"""

import os
import sys
import time

import numpy as np
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------- SETTINGS ----------
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
CHUNK_WORDS = 200      # words per chunk
CHUNK_OVERLAP = 40     # words shared between neighbouring chunks
TOP_K = 4              # how many chunks to retrieve per question
EMBED_BATCH = 50       # chunks sent per embedding request


# ---------- 0. CLIENT ----------
def get_client(api_key=None):
    """Create the Gemini client. Uses the GEMINI_API_KEY env variable if no key is passed."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No API key found. Set GEMINI_API_KEY or pass api_key.")
    return genai.Client(api_key=key)


# ---------- 1. LOAD ----------
def load_pdf(file):
    """Return a list of (page_number, text). `file` can be a path or an uploaded file object."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


# ---------- 2. CHUNK ----------
def chunk_pages(pages, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split each page into overlapping word windows, remembering the page number."""
    chunks = []
    step = chunk_words - overlap
    for page_num, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_words])
            if piece:
                chunks.append({"page": page_num, "text": piece})
            if start + chunk_words >= len(words):
                break
    return chunks


# ---------- 3. EMBED ----------
def _with_retry(fn, tries=5):
    """The free tier has rate limits, so wait and retry when we hit them."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if rate_limited and attempt < tries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise


def embed_texts(client, texts, task_type):
    """Turn a list of texts into a normalised (n x dims) numpy array."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        result = _with_retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        ))
        vectors.extend(e.values for e in result.embeddings)

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    # Normalise so a simple dot product equals cosine similarity
    return arr / norms


def build_index(client, chunks):
    """Embed every chunk. The 'index' is just the chunks + their vectors."""
    vectors = embed_texts(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    return {"chunks": chunks, "vectors": vectors}


# ---------- 4. RETRIEVE ----------
def retrieve(client, index, question, top_k=TOP_K):
    """Return the top_k most relevant chunks as (chunk, similarity_score)."""
    if not index["chunks"]:
        return []

    q_vec = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
    scores = index["vectors"] @ q_vec
    best = np.argsort(scores)[::-1][:top_k]
    return [(index["chunks"][i], float(scores[i])) for i in best]


# ---------- 5. GENERATE ----------
def answer(client, question, retrieved):
    """Ask Gemini to answer using only the retrieved chunks."""
    if not retrieved:
        return "I couldn't find that in the document."

    context = "\n\n".join(
        f"[Page {chunk['page']}]\n{chunk['text']}" for chunk, _ in retrieved
    )
    prompt = (
        "You are a helpful assistant answering questions about a document.\n"
        "Use ONLY the context below. If the answer is not in the context, say "
        "\"I couldn't find that in the document.\" Mention the page number(s) you used.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )
    response = _with_retry(lambda: client.models.generate_content(
        model=CHAT_MODEL, contents=prompt
    ))
    return response.text


# ---------- TERMINAL / COLAB VERSION ----------
def main(pdf_path):
    client = get_client()

    print(f"Reading {pdf_path} ...")
    pages = load_pdf(pdf_path)
    if not pages:
        print("No text found. This may be a scanned PDF (images only).")
        return

    chunks = chunk_pages(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks. Building index ...")
    index = build_index(client, chunks)
    print("Ready! Ask questions about the document (type 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        retrieved = retrieve(client, index, question)
        print("\nAssistant:", answer(client, question, retrieved))
        pages_used = sorted({c["page"] for c, _ in retrieved})
        print(f"(Sources: pages {pages_used})\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_core.py your_document.pdf")
    else:
        main(sys.argv[1])
