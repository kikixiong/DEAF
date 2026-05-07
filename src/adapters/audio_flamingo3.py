"""Adapter for Audio Flamingo 3 (NVIDIA).

AF3 uses NVIDIA's `llava` package shipped inside the Audio_Flamingo3 repo.
Clone the repo and point AF3_DIR (or code_dir in models.yaml) at the checkout.

See docs/models/audio_flamingo3.md for setup details.
"""

import os
import sys

from .base import BaseAdapter


class AudioFlamingo3Adapter(BaseAdapter):

    def load(self, model_cfg: dict):
        af3_dir = model_cfg.get("code_dir") or os.environ.get("AF3_DIR", "")
        if not af3_dir:
            raise ValueError(
                "Audio Flamingo 3 code dir required. "
                "Set code_dir in models.yaml or AF3_DIR env var. "
                "See docs/models/audio_flamingo3.md."
            )
        if af3_dir not in sys.path:
            sys.path.insert(0, af3_dir)

        import llava
        from llava import conversation as clib
        from llava.media import Sound

        model_path = model_cfg.get("model_path") or os.environ.get("AF3_MODEL", "")
        if not model_path:
            raise ValueError(
                "Audio Flamingo 3 model_path required. "
                "Set model_path in models.yaml or AF3_MODEL env var."
            )

        self.Sound = Sound
        self.model = llava.load(model_path)
        self.model = self.model.to("cuda:0")

        clib.default_conversation = clib.conv_templates["auto"].copy()
        print("[Audio Flamingo 3] Loaded on cuda:0")

    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        sound = self.Sound(audio_path)
        response = self.model.generate_content([sound, prompt])
        return response.strip()
