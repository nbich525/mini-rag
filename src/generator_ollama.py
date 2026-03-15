import os, requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3")

def generate_answer(query, contexts, max_tokens=256):
    context_text = "\n\n---\n".join([f"Doc {cid}:\n{txt}" for cid, txt, _ in contexts])
    prompt = f"Use the contexts to answer. If not found, say 'I don't know'.\n\n{context_text}\n\nQuestion: {query}\nAnswer:"
    payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False
    }
    url = f"{OLLAMA_URL}/api/generate"
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "response" in data:
            return data["response"]
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return str(data)
    except Exception as e:
        return " ".join([txt for _,txt,_ in contexts])[:800] + f"\n\n[OLLAMA_ERR:{e}]"