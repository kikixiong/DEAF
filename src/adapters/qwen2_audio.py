"""Adapter for Qwen2-Audio-7B-Instruct.

Loads via transformers' Qwen2AudioForConditionalGeneration. Requires the
HuggingFace snapshot to be available locally (model_path in models.yaml,
or QWEN2_AUDIO_MODEL env var as fallback).
"""

import os

import librosa
import torch

from .base import BaseAdapter


class Qwen2AudioAdapter(BaseAdapter):

    def load(self, model_cfg: dict):
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        model_path = model_cfg.get("model_path") or os.environ.get("QWEN2_AUDIO_MODEL", "")
        if not model_path:
            raise ValueError(
                "Qwen2-Audio model_path required. "
                "Set model_path in models.yaml or QWEN2_AUDIO_MODEL env var."
            )

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.device = next(self.model.parameters()).device
        print(f"[Qwen2-Audio] Loaded on {self.device}")

    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        audio_data = librosa.load(audio_path, sr=self.sampling_rate)[0]
        inputs = self.processor(
            text=text, audios=[audio_data], return_tensors="pt",
            sampling_rate=self.sampling_rate, padding=True,
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        max_new = 10 if fmt == "mcq" else 128
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new)
            gen_ids = gen_ids[:, inputs["input_ids"].size(1):]

        response = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response.strip()
