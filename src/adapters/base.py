"""Base adapter interface for all DEAF models."""

from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """All model adapters must implement load() and infer()."""

    @abstractmethod
    def load(self, model_cfg: dict):
        """Load model into memory. Called once before any infer() calls."""
        ...

    @abstractmethod
    def infer(self, audio_path: str, prompt: str, fmt: str = "oe") -> str:
        """Run inference on a single audio file with the given prompt.

        Args:
            audio_path: Absolute path to the audio file.
            prompt:     The full text prompt sent to the model.
            fmt:        "mcq" for multiple-choice (short generation),
                        "oe" for open-ended (longer generation).

        Returns:
            The raw model response string.
        """
        ...

    def unload(self):
        """Optional cleanup. Default no-op."""
        pass
