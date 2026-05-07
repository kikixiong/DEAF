"""Adapter for SALMONN-7B.

SALMONN ships its model code in a separate repository; clone it first and
point SALMONN_DIR (or code_dir in models.yaml) at the checkout.

See docs/models/salmonn.md for setup details.
"""

import contextlib
import os
import sys
import tempfile

import librosa
import soundfile as sf
import torch

from .base import BaseAdapter


class SALMONNAdapter(BaseAdapter):

    def load(self, model_cfg: dict):
        salmonn_dir = model_cfg.get("code_dir") or os.environ.get("SALMONN_DIR", "")
        if not salmonn_dir:
            raise ValueError(
                "SALMONN code dir required. "
                "Set code_dir in models.yaml or SALMONN_DIR env var. "
                "See docs/models/salmonn.md."
            )
        if salmonn_dir not in sys.path:
            sys.path.insert(0, salmonn_dir)

        from config import Config
        from models.salmonn import SALMONN
        from transformers import WhisperFeatureExtractor

        cfg_path = model_cfg.get(
            "cfg_path",
            os.path.join(salmonn_dir, "configs", "decode_config.yaml"),
        )

        class _Args:
            def __init__(self):
                self.cfg_path = cfg_path
                self.options = None

        self.cfg = Config(_Args())
        self.model = SALMONN.from_config(self.cfg.config.model)
        self.model.cuda()
        self.model.eval()

        # Patch maybe_autocast to use bfloat16 (fp16 causes conv1d mismatch).
        def _maybe_autocast(dtype=torch.bfloat16):
            if self.model.device != torch.device("cpu"):
                return torch.cuda.amp.autocast(dtype=dtype)
            return contextlib.nullcontext()
        self.model.maybe_autocast = _maybe_autocast

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(
            self.cfg.config.model.whisper_path
        )

        from utils import prepare_one_sample
        self._prepare_one_sample = prepare_one_sample

        print("[SALMONN] Loaded")

    def _downsample(self, wav_path: str, target_sr: int = 16000) -> str:
        audio, orig_sr = librosa.load(wav_path, sr=None)
        if orig_sr == target_sr:
            return wav_path
        audio_rs = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio_rs, target_sr)
        return tmp.name

    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        processed = self._downsample(audio_path)
        try:
            samples = self._prepare_one_sample(processed, self.wav_processor)
            formatted = [
                self.cfg.config.model.prompt_template.format(
                    "<Speech><SpeechHere></Speech> " + prompt.strip()
                )
            ]
            with torch.no_grad():
                output = self.model.generate(
                    samples, self.cfg.config.generate, prompts=formatted
                )[0]
            output = output.strip()
            for tag in ("<s>", "</s>"):
                output = output.replace(tag, "").strip()
            return output
        finally:
            if processed != audio_path:
                os.unlink(processed)
