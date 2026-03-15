import os, pickle, numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data", "documents.pdf")
OUT = os.path.join(BASE, "..", "vectorstore")
os.makedirs(OUT, exist_ok=True)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_pdf(path):
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def chunk_text(text):
    words = text.split()
    chunks=[]
    i=0
    while i < len(words):
        chunk = " ".join(words[i:i+CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build():
    assert os.path.exists(DATA), f"Place a small PDF at {DATA}"
    print("Loading PDF...")
    text = load_pdf(DATA)
    chunks = chunk_text(text)
    print(f"Chunks: {len(chunks)}")

    print("Loading embedder...")
    embedder = SentenceTransformer(EMB_MODEL)
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(OUT, "faiss.index"))

    with open(os.path.join(OUT, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    np.save(os.path.join(OUT, "embeddings.npy"), embeddings)

    # BM25
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(OUT, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print("Ingest finished.")

if __name__=="__main__":
    build()