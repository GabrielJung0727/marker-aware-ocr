import os
from typing import Any, Dict, List
import requests


class OllamaClient:
    def __init__(self, host: str) -> None:
        self.host = host.rstrip('/')

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
        output_format: str | None = None,
    ) -> str:
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if output_format:
            payload["format"] = output_format
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
        output_format: str | None = None,
    ) -> str:
        url = f"{self.host}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if output_format:
            payload["format"] = output_format
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {}) or {}
        return str(message.get("content", ""))


def get_default_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")
