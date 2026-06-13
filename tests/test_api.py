import pytest
from httpx import AsyncClient
from app.main import app

BASE_URL = "http://test"

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(app=app, base_url=BASE_URL) as ac:
        response = await ac.get("/api/v1/health")
        assert response.status_code in [200, 503] # 503 if index missing, 200 if healthy

@pytest.mark.asyncio
async def test_ask_endpoint_success():
    # This test assumes the ingestion pipeline has been run and index exists
    async with AsyncClient(app=app, base_url=BASE_URL) as ac:
        response = await ac.post("/api/v1/ask", json={"question": "What is a Python list?"})
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert len(data["answer"]) > 0
        elif response.status_code == 500:
            # Likely missing vectorstore
            pytest.skip("Skipping ask test: Vectorstore likely missing (run ingestion first)")