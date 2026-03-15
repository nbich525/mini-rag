import os, pickle, numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(__file__)
VEC = os.path.join(BASE, "..", "vectorstore")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_store():
    with open(os.path.join(VEC, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    bm25 = pickle.load(open(os.path.join(VEC, "bm25.pkl"), "rb"))
    index = faiss.read_index(os.path.join(VEC, "faiss.index"))
    embeddings = np.load(os.path.join(VEC, "embeddings.npy"))
    return chunks, bm25, index, embeddings

def rrf(rank_lists, k=60):
    agg={}
    for rl in rank_lists:
        for pos, doc in enumerate(rl, start=1):
            agg[doc]=agg.get(doc,0)+1.0/(pos+k)
    return [doc for doc,_ in sorted(agg.items(), key=lambda x:x[1], reverse=True)]

class HybridRetriever:
    def __init__(self):
        self.chunks, self.bm25, self.index, self.emb = load_store()
        self.embedder = SentenceTransformer(EMB_MODEL)

    def retrieve(self, query, top_k_bm25=10, top_k_dense=10, final_k=10):
        # BM25
        tokenized = query.lower().split()
        bm25_texts = self.bm25.get_top_n(tokenized, self.chunks, n=top_k_bm25)
        bm25_ids = [self.chunks.index(t) for t in bm25_texts]
        # dense
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D,I = self.index.search(q_emb, top_k_dense)
        dense_ids = I[0].tolist()
        fused = rrf([bm25_ids, dense_ids], k=60)
        top = fused[:final_k]
        return [(i, self.chunks[i]) for i in top]