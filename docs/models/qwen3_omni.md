# Qwen3-Omni

`Qwen/Qwen3-Omni-30B-A3B-Instruct` served by SiliconFlow.

## 1. Get a key

| Region        | Site                         | Notes |
|---------------|------------------------------|-------|
| Mainland China | https://siliconflow.cn       | Currently the only place that exposes Qwen3-Omni-30B at the time of writing |
| International  | https://siliconflow.com      | Qwen3-Omni was disabled internationally in April 2026; check before relying on it |

```bash
export SILICONFLOW_API_KEY=sk-...
```

## 2. (Optional) override the base URL

The default is `https://api.siliconflow.com/v1`. If you have an mainland-only
key, switch the base in `configs/models.yaml`:

```yaml
- name: qwen3-omni
  type: api
  adapter: qwen3_omni
  model_id: Qwen/Qwen3-Omni-30B-A3B-Instruct
  base_url: https://api.siliconflow.cn/v1     # mainland endpoint
  api_key: ${SILICONFLOW_API_KEY}
  max_tokens: 512
```

## 3. Run

```bash
python src/run_inference.py \
    --model qwen3-omni \
    --eval-csv data/prompts/esc_eval.csv \
    --workers 4         # SiliconFlow is rate-limited; keep workers low
```

## Notes

- Qwen3-Omni accepts audio as a base64 `data:` URI (different from OpenAI's
  `input_audio` block). The adapter handles the conversion.
- 30B-A3B is an MoE; latency is comparable to the 7B Gemini Flash models for
  short prompts.
