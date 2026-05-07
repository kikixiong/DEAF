"""Adapter for DeSTA2-8B-beta.

DeSTA2 ships the `desta` Python package inside its evaluation repo; clone
that repo and point DESTA_DIR (or code_dir in models.yaml) at the checkout.

See docs/models/desta2.md for setup details.
"""

import os
import sys

from .base import BaseAdapter


class DeSTA2Adapter(BaseAdapter):

    def load(self, model_cfg: dict):
        desta_dir = model_cfg.get("code_dir") or os.environ.get("DESTA_DIR", "")
        if not desta_dir:
            raise ValueError(
                "DeSTA2 code dir required. "
                "Set code_dir in models.yaml or DESTA_DIR env var. "
                "See docs/models/desta2.md."
            )
        if desta_dir not in sys.path:
            sys.path.insert(0, desta_dir)

        from desta import DestaModel

        model_path = model_cfg.get("model_path") or os.environ.get("DESTA2_MODEL", "")
        if not model_path:
            raise ValueError(
                "DeSTA2 model_path required. "
                "Set model_path in models.yaml or DESTA2_MODEL env var."
            )

        self.model = DestaModel.from_pretrained(model_path, local_files_only=True)
        self.model.to("cuda")
        print("[DeSTA2] Loaded on cuda")

    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        messages = [
            {"role": "system", "content": "You are a helpful voice assistant."},
            {"role": "audio", "content": audio_path},
            {"role": "user", "content": prompt},
        ]

        max_new = 10 if fmt == "mcq" else 128
        generated_ids = self.model.chat(messages, max_new_tokens=max_new, do_sample=False)
        response = self.model.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response.strip()
