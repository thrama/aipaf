# AIPAF Agent

**AI Project Assessment Framework — Agent CLI**

[![tests](https://github.com/thrama/aipaf/actions/workflows/tests.yml/badge.svg)](https://github.com/thrama/aipaf/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Strumento per condurre assessment strutturati di progetti AI secondo il framework AIPAF,
basato su NIST AI RMF 1.0, ISO/IEC 42001:2023 e EU AI Act.

> **English summary.** AIPAF is a Python CLI and methodology for assessing AI projects
> against NIST AI RMF 1.0, ISO/IEC 42001:2023 and the EU AI Act. It scores a project on six
> dimensions (strategic alignment, risk classification, data readiness, technical feasibility,
> ethical & responsible AI, operational readiness) with a deterministic scoring engine and a
> GREEN / YELLOW / RED approval gate. An LLM (Anthropic Claude or a local Ollama model) is used
> only to structure interview answers and to write narratives; the numbers never come from the
> model. An optional RAG layer (ChromaDB) injects the relevant regulatory text into prompts.
> Documentation and CLI are in Italian; the code, tests and configuration are self-explanatory.

> **Disclaimer.** AIPAF è uno strumento di supporto metodologico. I suoi punteggi e gate
> **non costituiscono consulenza legale**, né una valutazione di conformità ai sensi
> dell'AI Act, né una certificazione ISO/IEC 42001. La classificazione di rischio prodotta
> è una stima basata sulle risposte fornite e va verificata da chi ha la responsabilità
> del progetto. Vedi anche [Sicurezza e dati](#sicurezza-e-dati).

---

## Installazione

```bash
# Clone del repository
git clone https://github.com/thrama/aipaf.git
cd aipaf

# Installazione base
pip install -e .

# Con supporto Claude
pip install -e ".[claude]"

# Con strumenti di sviluppo
pip install -e ".[dev]"

# Per il notebook AIPAF_Assessment.ipynb (jupyter, pandas, matplotlib, seaborn, ipywidgets)
pip install -e ".[notebook]"
python -m ipykernel install --user --name aipaf --display-name "Python (aipaf)"
```

Con `uv`: `uv venv --python 3.12`, poi `uv pip install -e ".[claude,dev,notebook]"`.
Le dipendenze sono definite solo in `pyproject.toml` (extra `claude`, `dev`, `notebook`).

---

## Configurazione

### Variabili d'ambiente

```bash
cp .env.example .env
# Edita .env con i tuoi valori
```

Il file `.env` nella cartella corrente viene caricato automaticamente da CLI e notebook
(le variabili già presenti nell'ambiente hanno la precedenza).

| Variabile | Default | Uso |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Obbligatoria con `--llm claude` |
| `AIPAF_CLAUDE_MODEL` | `claude-sonnet-4-6` | Modello Claude, senza toccare il codice |
| `AIPAF_CLAUDE_TEMPERATURE` | `auto` | `auto`: passa temperature e la disabilita se il modello la rifiuta; `off`: non la passa mai |
| `OLLAMA_HOST` | `http://localhost:11434` | Endpoint Ollama |
| `AIPAF_CONFIG_PATH` | JSON impacchettato | Framework personalizzato (vedi sotto) |

### Claude (Anthropic API)

```bash
# Linux/macOS
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### Ollama (locale)

Installa Ollama da https://ollama.com, poi scarica i modelli:

```bash
# Modello LLM consigliato per notebook 16 GB RAM
ollama pull qwen3.5:4b        # 2.7 GB

# Modello embedding per il RAG layer
ollama pull nomic-embed-text  # 274 MB
```

> Se `ollama pull` fallisce per problemi di rete, scarica i file `.gguf`
> da [HuggingFace](https://huggingface.co) e importali con `ollama create <nome> -f Modelfile`.

---

## Utilizzo

### Verifica configurazione

```bash
aipaf check --llm ollama --model qwen3.5:4b
aipaf check --llm claude
```

### Gestione documenti RAG

Aggiungi i documenti nelle cartelle appropriate:

```
docs/sources/regulatory/    ← EU AI Act, NIST AI RMF, ISO 42001, ...
docs/sources/company/       ← codice etico aziendale, policy interne, ... (non versionata)
```

EU AI Act e NIST AI RMF 1.0 sono inclusi nel repository (documenti pubblici); ISO/IEC 42001
è uno standard a pagamento e **non è incluso**: va acquistato e copiato in `regulatory/`.
Dettagli e fonti in [`docs/sources/regulatory/README.md`](docs/sources/regulatory/README.md).

Poi indicizza:

```bash
# Prima indicizzazione (o dopo aver aggiunto nuovi file)
aipaf rag ingest --backend ollama

# Forza re-ingestione di tutti i file
aipaf rag ingest --backend ollama --force

# Verifica stato indice
aipaf rag status
```

### Assessment interattivo

```bash
# Con Ollama
aipaf assess --llm ollama --model qwen3.5:4b

# Con Ollama + RAG (contesto normativo iniettato nei prompt)
aipaf assess --llm ollama --model qwen3.5:4b --rag

# Con Claude (qualità superiore — consigliato per progetti HIGH RISK)
aipaf assess --llm claude

# Solo alcune dimensioni (re-assessment parziale → gate PARTIAL, vedi "Semantica dello scoring")
aipaf assess --llm claude --dims D2,D3,D5

# Con salvataggio diretto del report .md
aipaf assess --llm claude --report output/report.md
```

### Generazione report

```bash
# Genera report .md da una sessione salvata
aipaf report assessment_progetto.json

# Con percorso output esplicito
aipaf report assessment_progetto.json --output output/report.md

# Senza narrative LLM (solo scoring deterministico)
aipaf report assessment_progetto.json --no-narrative
```

### Scoring batch da file JSON

```bash
# Calcola score da risposte pre-compilate
aipaf score risposte.json --llm claude

# Senza narrative LLM
aipaf score risposte.json --no-narrative
```

Formato del file di input:

```json
{
  "intake": { "project_name": "...", "sector": "banking_finance", "eu_ai_act_level": "HIGH_RISK",
              "personal_data_treated": false, "...": "..." },
  "responses": [
    { "criterion_id": "D1.C1", "score": 2, "confidence": "M", "evidence": "..." },
    { "criterion_id": "D2.C2", "not_applicable": true, "evidence": "Nessun dato personale trattato" }
  ],
  "assessed_dimensions": ["D1", "D2"]
}
```

`assessed_dimensions` è opzionale: se assente, una dimensione è considerata valutata
se ha almeno una risposta. Un `criterion_id` non presente nel framework fa fallire lo scoring
(errore esplicito, non viene ignorato).

---

## Semantica dello scoring

**Criteri non applicabili (N/A).** Durante `assess` si risponde `na` a un criterio (con
motivazione); in batch si usa `"not_applicable": true`. Il criterio **esce dal denominatore**
della dimensione: la media pesata è calcolata sui soli criteri applicabili, quindi un N/A non
conta né a favore né contro. Se tutti i criteri di una dimensione sono N/A, la dimensione è
esclusa da composito e gate con nota esplicita nel report.
Le `skip_condition` del config (es. `personal_data_treated == false` su D2.C2 e D3.C4) sono
valutate contro l'intake: se vere, la CLI propone l'N/A con conferma.

**Criteri saltati (`skip`).** Diverso da N/A: il criterio *si applica* ma non è stato valutato.
Viene registrato con confidence L (score effettivo 0, flag *data needed*) e **pesa** sulla
dimensione. È la scelta conservativa voluta dal framework.

**Assessment parziale (PARTIAL).** Con `--dims` (o risposte solo per alcune dimensioni) le
dimensioni non valutate sono escluse da composito, gate e pattern alert. Il gate risultante è
`PARTIAL`, accompagnato da un `provisional_gate` calcolato sulle sole dimensioni valutate.
Un assessment parziale non produce mai un gate GREEN/YELLOW/RED definitivo.

**Fallimenti LLM.** Un output non interpretabile (JSON malformato, settore o livello EU AI Act
fuori dominio, score fuori scala) viene ritentato una volta e poi solleva `LLMOutputError`.
Nessun valore di default silenzioso. In `assess`:
- intake fallito → l'assessment si ferma (settore e livello EU AI Act condizionano pesi e flag);
- criterio fallito → registrato con confidence L, warning in console, elenco a fine sessione.

**Tracciabilità.** La sessione e il report registrano provider e modello LLM che hanno
interpretato le risposte (`llm_provider`, `llm_model`) e la versione di AIPAF.

**Soglie.** `DIMENSION_CRITICAL_THRESHOLD` (0.5), `GATE_YELLOW_THRESHOLD` (1.0),
`GATE_GREEN_THRESHOLD` (2.0) vivono in `aipaf/models.py`, unica fonte di verità per engine e report.

### Riepilogo sessione salvata

```bash
aipaf summary assessment_progetto.json
```

---

## Struttura progetto

```
aipaf/
├── aipaf/                            # Package Python
│   ├── __init__.py
│   ├── agent.py                      # Orchestrazione LLM + engine + RAG
│   ├── cli.py                        # CLI Typer (assess, report, score, summary, check, rag)
│   ├── engine.py                     # Scoring Engine deterministico (zero dipendenze AI)
│   ├── models.py                     # Pydantic models
│   ├── report.py                     # Generazione report Markdown
│   ├── config/
│   │   └── AIPAF_framework_config.json  # Knowledge base del framework (impacchettata)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                   # LLMProvider (ABC), Message, LLMResponse
│   │   ├── claude_provider.py
│   │   ├── ollama_provider.py
│   │   └── factory.py
│   └── rag/
│       ├── __init__.py
│       ├── ingestor.py               # PDF/DOCX/TXT → chunk → ChromaDB
│       └── retriever.py              # Query interface + context_for_criterion()
├── docs/
│   ├── sources/
│   │   ├── regulatory/               # EU AI Act, NIST AI RMF (ISO 42001 da aggiungere a mano)
│   │   └── company/                  # Codice etico, policy interne (non versionata)
│   ├── index/                        # ChromaDB (auto-generato, non versionato)
│   ├── archive/                      # Versioni precedenti dei documenti (non versionata)
│   ├── manifest.json                 # Checksum documenti indicizzati (auto-generato, non versionato)
│   └── LICENSE-DOCS.md               # Licenza CC BY 4.0 dei documenti del framework
├── output/                           # Report .md generati (non versionati)
├── tests/
│   ├── test_engine.py                # Gate logic, pesi settoriali, pattern alerts, N/A, PARTIAL
│   ├── test_agent.py                 # Agent con provider mock: retry, errori, statelessness
│   ├── test_llm.py                   # Provider mock, factory, _parse_json, Claude/Ollama unit
│   ├── test_rag.py                   # Ingestor, retriever, HashEmbeddingFunction
│   └── test_report.py                # ReportGenerator, struttura Markdown, salvataggio
├── .github/workflows/tests.yml       # CI: pytest su Python 3.11/3.12, ruff, build wheel
├── .env.example
├── .gitattributes
├── .gitignore
├── .pre-commit-config.yaml           # ruff + nbstripout
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                           # MIT (codice)
├── SECURITY.md
├── pyproject.toml
└── README.md
```

---

## Architettura

**Scoring Engine** (`engine.py`) — completamente deterministico, zero dipendenze AI.
Calcola score, gate, pattern alerts e normative flags. A parità di input produce sempre lo
stesso output. Il config JSON è impacchettato nel wheel (`aipaf/config/`) e caricato via
`importlib.resources`; `AIPAF_CONFIG_PATH` permette di sostituirlo. Le sezioni
`approval_gates`, `scoring_engine` ed `eu_ai_act_classification` del JSON sono descrittive:
le soglie operative sono in `models.py`.

**LLM Layer** (`llm/`) — interfaccia astratta `LLMProvider` con implementazioni per
Claude (API Anthropic) e Ollama (locale). Aggiungere un provider richiede solo una nuova
classe e una riga in `factory.py`.

**RAG Layer** (`rag/`) — indicizza i documenti normativi e aziendali in ChromaDB e li
inietta come contesto nei prompt dell'agent. Usa `OllamaEmbeddingFunction` in produzione
e `HashEmbeddingFunction` (deterministica, zero download) nei test. I documenti vengono
re-indicizzati solo se il checksum è cambiato.

**Agent** (`agent.py`) — usa l'LLM per strutturare intake e risposte in `CriterionResponse`,
arricchisce i prompt con contesto RAG (opzionale), delega il calcolo numerico all'engine.
Le chiamate strutturate sono stateless (ogni criterio è valutato senza il contesto dei
precedenti); lo storico conversazionale serve solo a executive summary e remediation plan.

**Report** (`report.py`) — genera un `.md` strutturato con scorecard ASCII, tabella
criteri per dimensione, executive summary, remediation plan e appendice normativa.

---

## Qualità LLM per task AIPAF

| Provider | Modello         | Ragionamento normativo | Note                                  |
| -------- | --------------- | :--------------------: | ------------------------------------- |
| Claude   | claude-sonnet-4-6 |         ★★★★★          | Consigliato per progetti HIGH RISK    |
| Ollama   | qwen3.5:9b      |         ★★★☆☆          | Punto dolce qualità/peso su 16 GB RAM |
| Ollama   | qwen3.5:4b      |         ★★☆☆☆          | Sviluppo e test, notebook 8-16 GB     |

> Il RAG layer migliora sensibilmente la qualità dei modelli locali iniettando
> il testo normativo rilevante direttamente nel prompt.

---

## Eseguire i test

```bash
# Tutti i test
python -m pytest tests/ -v

# Solo engine e agent (nessuna dipendenza esterna, nessun LLM)
python -m pytest tests/test_engine.py tests/test_agent.py -v

# Solo RAG (usa HashEmbeddingFunction, Ollama non richiesto)
python -m pytest tests/test_rag.py -v

# Solo report
python -m pytest tests/test_report.py -v

# Con coverage
python -m pytest tests/ --cov=aipaf --cov-report=term-missing
```

---

## Sicurezza e dati

- Con `--llm claude` le risposte dell'intervista e gli estratti dei documenti RAG vengono
  inviati all'API Anthropic; con `--llm ollama` tutto resta sulla macchina locale. Scegli il
  provider in base alla riservatezza del progetto valutato.
- Sessioni e report (`output/`, `assessment_*.json`) e i documenti in
  `docs/sources/company/` sono esclusi dal versionamento: non committarli.
- Per segnalare una vulnerabilità vedi [SECURITY.md](SECURITY.md).

---

## Licenza

- **Codice**: [MIT](LICENSE).
- **Documenti del framework** (Framework, Questionario, parte descrittiva del config JSON):
  [CC BY 4.0](docs/LICENSE-DOCS.md).
- I testi normativi di terze parti restano soggetti alle rispettive condizioni
  (vedi `docs/sources/regulatory/README.md`).

---

## Autore

**Lorenzo Lombardi** — Principal Data Architect. Il progetto è sviluppato e mantenuto a
titolo personale.

Contributi e segnalazioni: vedi [CONTRIBUTING.md](CONTRIBUTING.md).
