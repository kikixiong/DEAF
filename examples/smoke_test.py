"""Minimal end-to-end smoke test for the DEAF API adapters.

What it does:
  1. Verifies every adapter module under src/adapters/ can be imported.
  2. If OPENROUTER_API_KEY is set, runs one real inference call against
     gpt-4o-audio, gemini-2.5-flash and gemini-3-flash on AUDIO_PATH and
     prints the response.
  3. If SILICONFLOW_API_KEY is set, additionally exercises qwen3-omni.

What it does NOT do:
  - Load any local-GPU model (qwen2-audio, salmonn, desta2,
    audio-flamingo-3). Those need their own conda env + downloaded weights;
    see ../docs/models/.

Usage:
    export OPENROUTER_API_KEY=sk-or-v1-...
    python examples/smoke_test.py path/to/some.wav

If no audio path is given, the script picks the first .wav under
$DEAF_DATA_DIR (if set) so you can run it without arguments once the data
is in place.
"""

import importlib
import os
import sys
import time
import traceback
from pathlib import Path


def find_default_audio() -> str | None:
    """Try to locate any wav under $DEAF_DATA_DIR; return None if not found."""
    root = os.environ.get("DEAF_DATA_DIR")
    if not root:
        return None
    for wav in Path(root).rglob("*.wav"):
        return str(wav)
    return None


def main():
    # Make src/ importable regardless of where we're invoked from.
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    # 1. Import smoke: every adapter must import cleanly.
    print("=" * 60)
    print("Import smoke test")
    print("=" * 60)
    adapters = [
        "adapters.base",
        "adapters.gpt4o",
        "adapters.gemini",
        "adapters.qwen3_omni",
        "adapters.qwen2_audio",
        "adapters.salmonn",
        "adapters.desta2",
        "adapters.audio_flamingo3",
    ]
    for mod_path in adapters:
        try:
            importlib.import_module(mod_path)
            print(f"  OK   import {mod_path}")
        except Exception as e:
            print(f"  WARN import {mod_path}: {type(e).__name__}: {e}")
            print(f"       (this is expected if the corresponding heavy "
                  f"deps such as torch / transformers are not installed)")

    # 2. Pick an audio file.
    audio = sys.argv[1] if len(sys.argv) > 1 else find_default_audio()
    if not audio or not os.path.isfile(audio):
        print()
        print("No audio file provided and none discovered under $DEAF_DATA_DIR.")
        print("Pass one explicitly to run the inference half:")
        print("    python examples/smoke_test.py path/to/some.wav")
        sys.exit(0)

    prompt = (
        "Listen carefully to the speaker's voice. "
        "Describe the emotion conveyed by the speaker (one word)."
    )
    print()
    print("=" * 60)
    print(f"Inference smoke test on: {audio}")
    print(f"Prompt: {prompt!r}")
    print("=" * 60)

    tests = []
    if os.environ.get("OPENROUTER_API_KEY"):
        tests += [
            ("gpt-4o-audio",     "adapters.gpt4o",  "GPT4oAdapter",
                {"model_id": "openai/gpt-4o-audio-preview",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key":  os.environ["OPENROUTER_API_KEY"],
                 "max_tokens": 64}),
            ("gemini-2.5-flash", "adapters.gemini", "GeminiAdapter",
                {"model_id": "google/gemini-2.5-flash",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key":  os.environ["OPENROUTER_API_KEY"],
                 "max_tokens": 64}),
            ("gemini-3-flash",   "adapters.gemini", "GeminiAdapter",
                {"model_id": "google/gemini-3-flash-preview",
                 "base_url": "https://openrouter.ai/api/v1",
                 "api_key":  os.environ["OPENROUTER_API_KEY"],
                 "max_tokens": 64}),
        ]
    else:
        print("  OPENROUTER_API_KEY not set - skipping OpenRouter adapters.")

    if os.environ.get("SILICONFLOW_API_KEY"):
        tests.append(
            ("qwen3-omni", "adapters.qwen3_omni", "Qwen3OmniAdapter",
                {"model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                 "base_url": "https://api.siliconflow.com/v1",
                 "api_key":  os.environ["SILICONFLOW_API_KEY"],
                 "max_tokens": 64})
        )
    else:
        print("  SILICONFLOW_API_KEY not set - skipping qwen3-omni.")

    if not tests:
        print()
        print("No API keys exported, nothing to call. Set OPENROUTER_API_KEY at minimum.")
        sys.exit(0)

    results = []
    for name, mod_path, cls_name, cfg in tests:
        print()
        print(f"--- {name} ---")
        try:
            cls = getattr(importlib.import_module(mod_path), cls_name)
            adapter = cls()
            adapter.load(cfg)
            t0 = time.time()
            response = adapter.infer(audio, prompt, fmt="oe")
            dt = time.time() - t0
            print(f"  ({dt:.1f}s) {response[:200]}")
            results.append((name, "OK", dt))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            results.append((name, "FAIL", 0.0))

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, status, dt in results:
        print(f"  {name:24} {status:6} {dt:5.1f}s")


if __name__ == "__main__":
    main()
