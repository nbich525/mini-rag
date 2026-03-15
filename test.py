from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_smoke():
    response = client.post("/ask", json={"q": "What is the main idea?"})
    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)

    print("Smoke test passed")
    print(data)

if __name__ == "__main__":
    test_smoke()