# agent/llm.py
import os
import requests
from typing import Optional

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"
MODEL = "phi4-mini"

def call_llm(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 35,      # ↓ faster
            "temperature": 0.6,     # ↓ less rambling
            "top_p": 0.9,
        }
    }

    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=(5, 60)  # fast connect + read
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        return f"Model error: {e}"