# Audio Flamingo 3 (NVIDIA)

NVIDIA's AF3 uses NVIDIA's `llava` package, shipped inside the
Audio_Flamingo3 repo. Clone the repo and point `AF3_DIR` at the checkout.

Upstream: https://github.com/NVIDIA/audio-flamingo

## 1. Conda env + clone

```bash
conda create -n af3 python=3.10 -y
conda activate af3

git clone https://github.com/NVIDIA/audio-flamingo.git /path/to/Audio_Flamingo3
cd /path/to/Audio_Flamingo3
pip install -r requirements.txt
pip install -r /path/to/DEAF/requirements-local.txt
```

Watch out: AF3's `llava` is a fork of the original LLaVA package. Don't
install `llava` from PyPI in the same env.

## 2. Download weights

```bash
huggingface-cli download nvidia/audio-flamingo-3 \
    --local-dir /path/to/audio-flamingo-3
```

Disk: ~30 GB. GPU memory: ~24 GB at bf16; needs an A6000 / A100 / RTX 5090.

## 3. Configure

```bash
export AF3_DIR=/path/to/Audio_Flamingo3
export AF3_MODEL=/path/to/audio-flamingo-3
```

## 4. Run

```bash
conda activate af3
bash scripts/run_local_model.sh audio-flamingo-3
```

## Notes & gotchas

- The adapter rebinds `clib.default_conversation` to the `"auto"`
  conversation template before each load. Without this, AF3 inherits a
  template from a previous session and produces malformed outputs.
- AF3 tends to be the *outlier* on DEAF — it's the only model in the
  rebuttal cycle that didn't fully cascade on RAVDESS L2 (TDS ~ 45% vs
  90–100% for the others). If your numbers diverge, check your decode
  template first.
- `model.generate_content([sound, prompt])` — the order matters; `sound`
  (a `llava.media.Sound`) must come first.
