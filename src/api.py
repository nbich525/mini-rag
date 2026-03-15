from fastapi import FastAPI
from pydantic import BaseModel
from src.retriever import HybridRetriever
from src.reranker import Reranker

from src.generator_ollama import generate_answer
llm_backend = "ollama"

app = FastAPI()
retriever = HybridRetriever()
reranker = Reranker()

class Query(BaseModel):
    q: str

@app.post("/ask")
def ask(payload: Query):
    q = payload.q
    candidates = retriever.retrieve(q, top_k_bm25=10, top_k_dense=10, final_k=20)
    ranked = reranker.rerank(q, candidates, top_k=5)
    answer = generate_answer(q, ranked)
    return {"answer": answer, "sources": [{"id":cid,"score":float(score)} for cid,_,score in ranked], "llm_backend": llm_backend}