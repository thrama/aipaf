# Contribuire ad AIPAF

Grazie dell'interesse. Regole essenziali:

## Setup

```bash
git clone https://github.com/thrama/aipaf.git
cd aipaf
pip install -e ".[dev,claude]"
pre-commit install          # ruff + nbstripout + controlli di base
python -m pytest tests/ -v  # nessuna API key né Ollama richiesti
```

## Pull request

- Apri prima una issue per modifiche non banali (nuove dimensioni/criteri, cambi alla
  semantica di scoring o dei gate, nuovi provider LLM).
- Una PR = un tema. Test aggiornati o aggiunti; `pytest` verde in CI.
- Aggiungi una voce in `CHANGELOG.md` sotto la versione in lavorazione.
- Non modificare le soglie di scoring in più punti: vivono in `aipaf/models.py`
  (`DIMENSION_CRITICAL_THRESHOLD` e affini) e vanno importate, non duplicate.
- Gli errori dell'LLM devono restare visibili: niente fallback silenziosi su output
  malformati.

## Cosa non committare

- `.env`, output di assessment (`output/`, `*.scored.json`, `assessment_*.json`).
- Documenti in `docs/sources/company/` (policy o codici etici di aziende reali).
- Output delle celle del notebook (`nbstripout` li rimuove automaticamente).
- Testi normativi non ridistribuibili (es. ISO/IEC 42001, che è a pagamento).
