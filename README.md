# CAREBench

**C**ompliance · **A**ccuracy · **R**obustness **Bench**mark — run models on SOB tasks and score their JSON outputs.

Gold data is loaded from Hugging Face at runtime. Model outputs are written locally under `data/predictions/`. See **[data/README.md](data/README.md)** for dataset paths.

---

## Layout

```
CAREBench/
├── README.md
├── pyproject.toml
├── LICENSE
├── test_models.py            # run models (Ollama) → data/predictions/*.jsonl
├── sob_eval/                 # metrics library (v2.3)
├── scripts/
│   └── score_all.py          # score predictions → results/*.csv
├── tests/
│   └── test_metrics_v2.py    # unit tests for L1/L2/L3 formulas
├── data/
│   ├── README.md             # HF paths for extraction + repair gold
│   └── predictions/          # JSONL outputs (gitignored)
├── results/                  # aggregated scores (CSV)
└── docs/METHODOLOGY.md
```

---

## Workflow

```bash
pip install -e ".[dev]"

# 1. Run models (requires Ollama + HF token for gated datasets)
export OLLAMA_URL=http://localhost:11434
python test_models.py --model qwen3-8b

# 2. Score outputs against HF gold
python -m scripts.score_all --all

# 3. Verify metric formulas
pytest -q
```

---

## Datasets (Hugging Face)

| Task | Dataset |
|------|---------|
| Extraction & reasoning | [`interfaze-ai/sob`](https://huggingface.co/datasets/interfaze-ai/sob) — `data/test-00000-of-00001.parquet` |
| Repair & error detection | [`MoetezF/SOB-with-errors-injection`](https://huggingface.co/datasets/MoetezF/SOB-with-errors-injection) |

---

## Methodology

Three layers (L1 compliance, L2 accuracy, L3 robustness) with accuracy-gated L3. Full formulas: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## License

MIT — see [LICENSE](LICENSE).
