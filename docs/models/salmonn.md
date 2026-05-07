# SALMONN-7B

SALMONN ships its model code in a separate repository; you must clone it and
point `SALMONN_DIR` at the checkout before the adapter will load.

Upstream: https://github.com/bytedance/SALMONN

## 1. Conda env + clone

```bash
conda create -n salmon_env python=3.9 -y
conda activate salmon_env

git clone https://github.com/bytedance/SALMONN.git /path/to/SALMONN
cd /path/to/SALMONN
pip install -r requirements.txt
pip install -r /path/to/DEAF/requirements-local.txt
```

`pip install -r SALMONN/requirements.txt` pins versions of
`transformers`, `peft`, `bitsandbytes`, `tokenizers`, `soundfile`,
`librosa`, etc. Don't override these.

## 2. Download weights

SALMONN-7B requires three sets of weights:

| Component      | Source                                                       |
|----------------|--------------------------------------------------------------|
| LLM backbone   | `lmsys/vicuna-7b-v1.5`                                       |
| Speech encoder | `openai/whisper-large-v2`                                    |
| Audio encoder  | `MIT/ast-finetuned-audioset-10-10-0.4593`                    |
| LoRA + Q-Former | SALMONN-7B checkpoint from upstream releases                 |

Follow upstream README; place each checkpoint on disk and edit
`SALMONN/configs/decode_config.yaml` so each `*_path` points at the right
directory.

## 3. Configure

```bash
export SALMONN_DIR=/path/to/SALMONN
```

`configs/models.yaml` defaults `cfg_path` to
`${SALMONN_DIR}/configs/decode_config.yaml`. Override it via `cfg_path:` in
yaml if you keep configs elsewhere.

## 4. Run

```bash
conda activate salmon_env
bash scripts/run_local_model.sh salmonn
```

## Notes & gotchas

- The adapter monkey-patches `model.maybe_autocast` to use `bfloat16`
  instead of `float16`. `float16` triggers a `conv1d` dtype mismatch on at
  least PyTorch 2.0–2.2 + CUDA 12.
- Audio is force-resampled to 16 kHz before being fed to Whisper; the
  resampled WAV is written to a NamedTemporaryFile and cleaned up after
  each call.
- SALMONN's `prompt_template` already wraps the user text in
  `<Speech><SpeechHere></Speech>` — the adapter prepends this for you, so
  the prompt passed to `infer()` should be the plain task text.
- Decoding takes ~3-5 s per clip on an RTX 4090; ~9k clips per run takes
  10–14 hours.
