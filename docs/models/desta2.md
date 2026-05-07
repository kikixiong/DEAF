# DeSTA2-8B-beta

DeSTA2 ships the `desta` Python package inside its evaluation repo. You must
clone the repo and point `DESTA_DIR` at the checkout before the adapter
will load.

Upstream: https://github.com/kehanlu/DeSTA2

## 1. Conda env + clone

```bash
conda create -n desta2 python=3.10 -y
conda activate desta2

git clone https://github.com/kehanlu/DeSTA2.git /path/to/DeSTA2
cd /path/to/DeSTA2
pip install -r requirements.txt
pip install -r /path/to/DEAF/requirements-local.txt
```

## 2. Download weights

```bash
huggingface-cli download DeSTA-ntu/DeSTA2-8B-beta \
    --local-dir /path/to/DeSTA2-8B-beta
```

Disk: ~16 GB. GPU memory: ~17 GB at bf16.

## 3. Configure

```bash
export DESTA_DIR=/path/to/DeSTA2
export DESTA2_MODEL=/path/to/DeSTA2-8B-beta
```

## 4. Run

```bash
conda activate desta2
bash scripts/run_local_model.sh desta2
```

## Notes & gotchas

- The adapter passes a 3-message conversation: a system instruction, the
  audio (role `audio`), and the user prompt. DeSTA2 accepts the audio path
  directly — the `desta` package handles loading and feature extraction.
- `do_sample=False` for reproducible decoding; `temperature` is unused.
- DeSTA2 occasionally refuses for strong-emotion content (RLHF-trained
  refusal pattern). Those responses come back as plain text such as
  *"I cannot determine ..."* and are scored as O by the judge.
