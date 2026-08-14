# Pre-computed benchmark aggregates (metrics v2.3)

Regenerate with:

```bash
python -m scripts.score_all --all
python -m scripts.export_dashboard_data
```

| File | Description |
|------|-------------|
| `extraction_summary.csv` | L1/L2/L3 per model × mode (prompt_only, reasoning_assisted) |
| `repair_summary.csv` | Repair task layered metrics |
| `benchmark-data.json` | Exported payload for the React dashboard |
| `all_layered.json` | Nested JSON by model (extraction tasks) |
| `summary_metrics.csv` | Legacy repair live-metric summary |
