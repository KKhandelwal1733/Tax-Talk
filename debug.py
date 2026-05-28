import os
import requests
from base64 import b64encode
from dotenv import load_dotenv

load_dotenv()

PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Basic auth header
token = b64encode(f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()
headers = {
    "Authorization": f"Basic {token}",
    "Content-Type": "application/json",
}

# 1. Test auth
resp = requests.get(f"{HOST}/api/public/projects", headers=headers)
print(f"Auth check: {resp.status_code} → {resp.json()}")

# 2. Send a raw trace directly via HTTP
payload = {
    "batch": [
        {
            "id": "test-trace-001",
            "type": "trace-create",
            "body": {
                "id": "test-trace-001",
                "name": "direct-http-test",
                "input": {"query": "test"},
                "output": {"answer": "ok"},
                "tags": ["debug"],
            },
            "timestamp": "2026-05-27T00:00:00.000Z",
        }
    ]
}

resp = requests.post(f"{HOST}/api/public/ingestion", headers=headers, json=payload)
print(f"Ingestion: {resp.status_code} → {resp.json()}")