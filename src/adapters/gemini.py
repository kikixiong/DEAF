"""Adapter for Gemini Flash audio models via OpenRouter (OpenAI-compatible).

OpenRouter normalises Gemini's audio input to the same `input_audio` block
that OpenAI uses, so the wire format here is identical to gpt4o.py.
The model_id (e.g. "google/gemini-2.5-flash") comes from configs/models.yaml.
"""

import base64
import os

import requests

from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):

    def load(self, model_cfg: dict):
        self.api_key = model_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        self.model_id = model_cfg.get("model_id", "google/gemini-2.5-flash")
        self.base_url = model_cfg.get("base_url", "https://openrouter.ai/api/v1")
        self.max_tokens = model_cfg.get("max_tokens", 512)

        if not self.api_key:
            raise ValueError(
                "API key required. Set api_key in models.yaml or OPENROUTER_API_KEY env var."
            )

        print(f"[Gemini] Ready (model: {self.model_id}, base: {self.base_url})")

    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
        if ext not in ("wav", "mp3", "flac", "ogg", "m4a"):
            ext = "wav"

        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": ext},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "modalities": ["text"],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
