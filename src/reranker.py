from sentence_transformers import SentenceTransformer, util
import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Reranker:
    def __init__(self, device='cpu'):
        self.model = SentenceTransformer(MODEL, device=device)

    def rerank(self, query, candidates, top_k=5):
        texts = [txt for _, txt in candidates]
        q_emb = self.model.encode(query, convert_to_numpy=True)
        c_embs = self.model.encode(texts, convert_to_numpy=True)
        sims = util.cos_sim(q_emb, c_embs)[0].cpu().numpy()
        scored = list(zip([cid for cid,_ in candidates], texts, sims.tolist()))
        scored_sorted = sorted(scored, key=lambda x:x[2], reverse=True)
        return scored_sorted[:top_k]