# Evidence + Reddit clean-tag

Listen first: tag real talk from Reddit, portal FAQs, Google PAA / PASF / autocomplete, and Trends into one shelf. Prompt writing comes later from `clusters.csv`.

## Global evidence (all sources)

```bash
cd "src/GEO Final Prompt Creation "
PYTHONPATH=/Users/abhijeet/geo-prompt-system python3 run_evidence.py --skip-llm --self-check --limit 50
```

| Flag | Meaning |
|---|---|
| `--sources` | `reddit,portal_faq,paa,pasf,autocomplete,trends` (default omits reddit for faster Google pass) |
| `--paa` | default `google_people_also_ask.xlsx` |
| `--pasf` | default `google_people_also_search.xlsx` |
| `--autocomplete` | default `google_autocomplete_suggestions.xlsx` (`text` = suggestion only) |
| `--portal-faq` | default `portal_faq_sample.csv` |
| `--trends` | default `trends_sample.csv` |
| `--xlsx` | Reddit export |
| `--skip-llm` | Rules-synth only (no Ollama) |
| `--limit N` | Cap reddit + each question source |
| `--workers 1` | Safer for local Ollama |
| `--self-check` | Shape checks on included sources |
| `--reddit-self-check` | Reddit golden acceptance |

**Normalized shelf** (`data/clean_evidence.csv`):

`evidence_id, source_type, source_name, text, topic, persona, stage, area, project, seed`

Also: `evidence_audit.csv`, `clusters.csv`, `evidence_report.txt`. Reddit view (`clean_threads.csv` / `audit.csv` / `dropped.csv`) when `reddit` is in `--sources`.

`asbl_mentions.csv` is not part of this pipeline.

## Reddit-only (legacy)

```bash
PYTHONPATH=/Users/abhijeet/geo-prompt-system python3 clean_threads.py
```

Config: `config/tags.yaml`, `config/llm_prompt.txt`. Cache: `data/llm_cache.jsonl`.
