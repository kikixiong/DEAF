"""Adapter for Qwen3-Omni via SiliconFlow API.

Note (2026-04): the international SiliconFlow endpoint disabled this model
mid-rebuttal cycle. Use the China endpoint at https://api.siliconflow.cn/v1
if the default base_url returns 404.
"""

import base64
import os

import requests

from .base import BaseAdapter


class Qwen3OmniAdapter(BaseAdapter):

    def load(self, model_cfg: dict):
        self.api_key = model_cfg.get("api_key") or os.environ.get("SILICONFLOW_API_KEY", "")
        self.model_id = model_cfg.get("model_id", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.base_url = model_cfg.get("base_url", "https://api.siliconflow.com/v1")
        self.max_tokens = model_cfg.get("max_tokens", 512)

        if not self.api_key:
            raise ValueError(
                "SiliconFlow API key required. "
                "Set api_key in models.yaml or SILICONFLOW_API_KEY env var."
            )

        print(f"[Qwen3-Omni] Ready (model: {self.model_id})")

    def _audio_to_data_uri(self, audio_path: str) -> str:
        """Convert local audio file to base64 data URI."""
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
        }
        mime = mime_map.get(ext, "audio/wav")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        data_uri = self._audio_to_data_uri(audio_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_id,
            "messages": messages,
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
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"].strip()
