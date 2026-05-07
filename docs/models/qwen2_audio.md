# Qwen2-Audio-7B-Instruct

Local GPU model loaded via 🤗 Transformers.

## 1. Conda env

```bash
conda create -n qwen2_audio python=3.10 -y
conda activate qwen2_audio
pip install -r requirements-local.txt
pip install transformers>=4.45 accelerate bitsandbytes  # latest tested
```

## 2. Download weights

```bash
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct \
    --local-dir /path/to/Qwen2-Audio-7B-Instruct
```

Disk: ~17 GB. GPU memory at bf16: ~17 GB; fits on a single A6000 / RTX 4090
or larger.

## 3. Configure

```bash
export QWEN2_AUDIO_MODEL=/path/to/Qwen2-Audio-7B-Instruct
```

`configs/models.yaml` already references `${QWEN2_AUDIO_MODEL}`; no edits
needed.

## 4. Run

```bash
conda activate qwen2_audio
bash scripts/run_local_model.sh qwen2-audio
```

Or one CSV at a time:

```bash
python src/run_inference.py \
    --model qwen2-audio \
    --eval-csv data/prompts/esc_eval.csv \
    --workers 1
```

## Notes & gotchas

- Always `workers=1` for local GPU adapters — the model object isn't
  thread-safe and you'd just queue on the GPU anyway.
- Qwen2-Audio uses `librosa` to load the wav at the model's sampling rate.
  Make sure the file is decodable (`soundfile` may need `libsndfile` system
  package on minimal Docker images).
- The benchmark prompt sets `max_new_tokens=10` for MCQ rows and `128` for OE
  rows. Tune in `src/adapters/qwen2_audio.py` if you need longer responses.
- `TRANSFORMERS_OFFLINE=1` is set in the adapter to avoid an HF hub call on
  every load — the model must already be on disk at `model_path`.

## Known issues

- Qwen2-Audio refuses (or returns format-incompatible output) on audio
  formats it doesn't expect. The DEAF rebuttal cycle observed 100% Other
  Rate when feeding RAVDESS WAVs unchanged; resampling to 16 kHz mono PCM
  before inference fixes this.
