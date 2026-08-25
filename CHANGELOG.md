# Changelog — AIPAF Agent

Tutte le modifiche rilevanti del progetto. Formato ispirato a [Keep a Changelog](https://keepachangelog.com/it/1.0.0/).

## [0.2.7] — 2026-08-25

### Corretto

- **RAG ingestor: manifest e indice verificati insieme.** `ingest_file` saltava un documento
  se il checksum coincideva con `docs/manifest.json`, senza controllare che i chunk fossero
  davvero in ChromaDB. Con `docs/index/` cancellato (o un manifest proveniente da un altro
  ambiente) il RAG restava vuoto con status "aggiornato". Ora l'ingestor conta i chunk
  presenti per quel file e re-indicizza se mancano o non tornano con `chunk_count`,
  con un avviso esplicito. Nuovo test `test_reingest_when_index_lost_but_manifest_current`.

### Modificato

- CI su `uv` (`astral-sh/setup-uv`, cache abilitata, `uv build` per il wheel), in linea con
  l'ambiente di sviluppo locale.
- Lint azzerato: `ruff check` e' ora **bloccante** in CI (rimosso `continue-on-error`).
  Import inutilizzati rimossi nei test, istruzioni multiple su una riga separate,
  variabili `l` rinominate (`lvl`, `line`). Nessuna modifica funzionale.

## [0.2.6] — 2026-08-25

### Preparazione alla pubblicazione su GitHub

- Rimossa l'affiliazione aziendale da intestazioni dei moduli, banner CLI, notebook,
  `pyproject.toml`, config JSON e README: il progetto è sviluppato a titolo personale.
- Aggiunti `LICENSE` (MIT, codice) e `docs/LICENSE-DOCS.md` (CC BY 4.0, documenti del framework).
- README: abstract in inglese, disclaimer (nessuna consulenza legale / conformity assessment),
  sezione "Sicurezza e dati", licenze, badge CI, URL di clone.
- CI GitHub Actions (`.github/workflows/tests.yml`): pytest + coverage su Python 3.11 e 3.12,
  ruff (non bloccante), build del wheel; `dependabot.yml` mensile.
- `.pre-commit-config.yaml` (ruff, nbstripout, detect-private-key, large files);
  `pre-commit` e `nbstripout` negli extra `[dev]`.
- `.gitattributes`: fine riga normalizzati a LF, binari dichiarati, notebook escluso dai diff.
- `.gitignore`: `docs/sources/company/*`, `docs/archive/`, `docs/manifest.json`
  (auto-generato: se versionato, un clone con `index/` vuoto salterebbe l'ingestion).
- `docs/sources/{regulatory,company}/README.md`: fonti, licenze dei testi normativi
  (ISO/IEC 42001 non ridistribuibile), regola "nessun documento aziendale nel repo".
- `SECURITY.md`, `CONTRIBUTING.md`; `[project.urls]`, keywords e classifiers in `pyproject.toml`.
- Regole ruff pinnate a `E4,E7,E9,F` (i default variano tra versioni e romperebbero la CI).
- Notebook: output delle celle rimossi.
- Fixture `test_rag.py`: azienda fittizia al posto di un'azienda reale.

## [0.2.5] — 2026-08-25

### Notebook `AIPAF_Assessment.ipynb` riallineato alla 0.2

- Setup ridotto a due celle: import dal package installato (niente `sys.path`, niente ricerca
  di `config/`, che dopo la 0.2.0 non esiste piu' alla radice e faceva fallire la cella 4);
  provider con motivo del fallimento; RAG opzionale che riusa il backend persistito.
- Le sei celle-dimensione usano una sola funzione `assess_dimension()` con la stessa
  semantica della CLI: `""` = saltato (L), `"na"` / `"na: motivo"` = non applicabile,
  N/A automatico dalle `skip_condition` (elencate dopo l'intake), interpretazione LLM
  fallita → registrato L con avviso.
- Scoring via `agent.finalize()` con le dimensioni effettivamente eseguite: gate `PARTIAL`
  se ne manca qualcuna; radar e barre marcano `n/v` e `N/A` invece di disegnare 0.
- `REPO_ROOT` (mai definito, `NameError` al salvataggio) sostituito da `OUTPUT_DIR = output/`
  (gia' in `.gitignore`); 4.3 salva sessione JSON **e** report Markdown.
- Portfolio: colonna LLM, gate `PARTIAL`, dimensioni non valutate come vuote (non 0)
  in tabella, heatmap e trend.
- Tabella dei gate e dei casi N/A/skip nelle note finali; kernel `Python (aipaf)`.
- Eseguito headless end-to-end con provider scriptato: 0 errori.

### Aggiunto

- Extra `[notebook]` in `pyproject.toml` (jupyter, ipykernel, ipywidgets, matplotlib, pandas,
  seaborn): `requirements.txt` resta solo per compatibilita'.
- **`.env` caricato automaticamente** da CLI e notebook (`python-dotenv`, `override=False`):
  `.env.example` ora mantiene la promessa che faceva.
- Vincolo `anthropic>=1.0` nell'extra `[claude]`, coerente con il provider 0.2.3.

## [0.2.4] — 2026-08-25

### Corretto

- **`aipaf rag status` falliva su un indice creato con Ollama**: il default era
  `--backend hash` e ChromaDB 1.5 rifiuta una embedding function diversa da quella
  persistita ("Embedding function conflict"). Ora `rag status` e `assess --rag` non
  richiedono il backend: `DocumentIngestor(embedding_backend=None)` riusa la
  configurazione salvata nell'indice. `rag ingest` continua a volerlo (default `ollama`)
  perche' deve poterlo creare. Se il backend richiesto non coincide con quello persistito,
  l'errore spiega il rimedio invece di mostrare lo stack di ChromaDB.
- `rag status` mostra il backend embedding persistito.

## [0.2.3] — 2026-08-25

### Corretto

- **Claude con SDK `anthropic` >= 1.0**: `messages.create()` non accetta piu' `temperature`
  e solleva `TypeError` (non `BadRequestError`): il fallback "auto" della 0.2.0 non lo
  intercettava e ogni chiamata falliva. Ora intercetta entrambi. Con SDK >= 1.0 la
  temperature e' di fatto sempre disabilitata: le chiamate strutturate si affidano al
  prompt e alla validazione lato agent, come gia' avveniva per i modelli che la rifiutavano.

## [0.2.2] — 2026-08-25

### Corretto

- **`aipaf rag ingest --backend ollama` falliva su installazione pulita**: la
  `OllamaEmbeddingFunction` di ChromaDB 1.5.x importa il pacchetto `ollama`, che non era
  tra le dipendenze. Aggiunto `ollama>=0.4` a `pyproject.toml`.
- **`aipaf check` diceva solo "non raggiungibile"**: `health_check()` inghiottiva
  l'eccezione. Ora `LLMProvider.last_error` conserva il motivo e la CLI lo stampa
  (proxy, chiave non valida, modello inesistente, timeout: sono errori diversi con
  rimedi diversi).

## [0.2.1] — 2026-08-24

Patch al RAG layer, emersa dal check del repository con i documenti reali in `docs/sources/`.

### Corretto

- **Retriever muto con backend `hash`** — `HashEmbeddingFunction` non implementava
  l'interfaccia `chromadb.EmbeddingFunction` (ChromaDB 1.5.x richiede `embed_query`,
  `get_config`, `build_from_config`). Le query sollevavano `AttributeError`, che
  `RAGRetriever._query_collection` inghiottiva restituendo `[]`: `context_for_criterion`
  tornava sempre stringa vuota e il test lo accettava. Il backend `ollama` (produzione) non
  era affetto. Ora la classe estende `EmbeddingFunction`; i test verificano che le query
  restituiscano chunk e che l'indice persistito si riapra correttamente.
- **DOCX: tabelle ignorate** — `_extract_text_docx` leggeva solo `doc.paragraphs`. Nel
  framework AIPAF criteri, pesi, gate e pattern sono in tabella: del Framework v1.5 venivano
  indicizzati 11k caratteri su 21k, del Questionario 3k su 36k (6 chunk su 76). Ora
  l'estrazione segue l'ordine del documento e include le righe di tabella (celle separate
  da `|`, celle unite deduplicate). Dopo l'aggiornamento **re-ingerire i DOCX**
  (`aipaf rag ingest --force`).

### Note

- Nel `manifest.json` consegnato non risultava indicizzato `OJ_L_202401689_IT_TXT.pdf`
  (EU AI Act, 687k caratteri, 1151 chunk): il RAG per D2 non aveva il testo del
  Regolamento. Un `aipaf rag ingest --backend ollama` lo aggiunge.

## [0.2.0] — 2026-08-24

Release di consolidamento della governance dello scoring, a valle dell'audit del codice.
Nessuna modifica ai pesi, ai criteri o alle soglie del framework: cambia _come_ il tool
gestisce ciò che non sa, non ciò che misura.

### Comportamenti che cambiano (leggere prima di aggiornare)

- **Criteri N/A** — Nuovo flag `not_applicable` su `CriterionResponse` (CLI: risposta `na`).
  Il criterio esce dal denominatore della dimensione. Prima l'unico modo di "saltare" un
  criterio era `skip`, che vale confidence L e **penalizza** la dimensione. Un progetto senza
  dati personali che azzerava D2.C2 (25% di D2) e D3.C4 (20% di D3) ora non viene più
  penalizzato per criteri che non gli si applicano.
- **`skip_condition` operativa** — Le condizioni nel config sono ora valutate contro l'intake
  (`personal_data_treated == false` su D2.C2 e D3.C4); la CLI propone l'N/A con conferma.
  Sintassi: `<campo intake> == <valore>` | `!=`. Il vecchio riferimento `P0.7` non esisteva
  in nessun modello e non era mai valutato.
- **Gate `PARTIAL`** — Con `--dims`, o con risposte solo per alcune dimensioni, il gate è
  `PARTIAL` e `provisional_gate` riporta l'esito sulle sole dimensioni valutate. Prima le
  dimensioni assenti entravano a 0.0 e forzavano RED, facendo scattare anche i pattern
  alert PA1–PA4. Composito, gate e pattern ora considerano solo le dimensioni valutate.
- **Fallimenti LLM rumorosi** — `_parse_json` non restituisce più `{}`: solleva
  `LLMOutputError` dopo un retry. Settore, livello EU AI Act, score e confidence fuori
  dominio sono errori, non default. In `assess`: intake fallito → exit 2; criterio fallito →
  registrato confidence L con warning e riepilogo finale.
- **Agent stateless per intake e criteri** — Ogni chiamata strutturata parte da una
  conversazione vuota. Prima lo storico cresceva per tutta la sessione e la valutazione di
  D4.C3 "vedeva" quella di D1.C1. Lo storico resta solo per executive summary e remediation.
- **Soglie in un solo posto** — `DIMENSION_CRITICAL_THRESHOLD`, `GATE_YELLOW_THRESHOLD`,
  `GATE_GREEN_THRESHOLD`, `REMEDIATION_THRESHOLD` in `aipaf/models.py`. Rimossa la
  duplicazione fra `DimensionScore.is_critical` e `ScoringEngine`.
- **Validazione risposte batch** — Un `criterion_id` sconosciuto in `aipaf score` è un
  `ValueError`, non viene più ignorato in silenzio. I pesi dei criteri per dimensione sono
  validati a caricamento del config (devono sommare a 100).

### Aggiunto

- `AssessmentSession.llm_provider`, `llm_model`, `aipaf_version`: tracciabilità di quale
  modello ha interpretato le risposte, in sessione JSON, summary e footer del report.
- `AssessmentSession.assessed_dimensions`, `partial`, `provisional_gate`;
  `DimensionScore.assessed`, `applicable`, `not_applicable`.
- `ScoringEngine.candidate_not_applicable(intake)` ed `engine.evaluate_skip_condition()`.
- `ClaudeProvider`: modello da `AIPAF_CLAUDE_MODEL` (default `claude-sonnet-4-6`);
  gestione `temperature` in modalità `auto` (riprova senza se il modello la rifiuta) o `off`
  via `AIPAF_CLAUDE_TEMPERATURE`.
- `OllamaProvider`: `TimeoutError` esplicito (`ReadTimeout` non è sottoclasse di
  `ConnectionError` e prima emergeva come errore generico); `OLLAMA_HOST` da ambiente;
  `model_available()` a tre stati (server giù / modello assente / ok). `health_check` non
  stampa più: il messaggio lo dà la CLI.
- Intake: campo `user_count_estimate` richiesto all'LLM (prima restava sempre `None`).
- Report: banner di assessment parziale, righe N/A esplicite, riga LLM in intestazione,
  escape di `|` e newline nelle celle (una evidenza multi-riga spaccava la tabella).
- `tests/test_agent.py` (21 test con provider scriptato) e nuovi test su engine, llm e
  report. Suite: **133 test**.
- `aipaf --help` e header CLI mostrano la versione.

### Modificato

- Config JSON spostato in `aipaf/config/` e impacchettato nel wheel (`package-data`);
  caricamento via `importlib.resources`, override con `AIPAF_CONFIG_PATH`. Prima il percorso
  relativo a `engine.py` funzionava solo in editable install. **Rimuovere la vecchia
  cartella `config/` alla radice del repo.**
- `pyproject.toml`: versione dinamica da `aipaf.__version__`.
- `cli.py`: rimosso l'hack `sys.path.insert`; `click.edit` al posto di `typer.edit`
  (assente nelle versioni recenti di Typer) con fallback al prompt se non c'è un editor.
- Remediation plan: le dimensioni deboli sono separate da newline (`''.join` → `'\n'.join`).
- `.env.example` e `README.md` aggiornati (variabili, semantica N/A/PARTIAL, formato batch).

### Note per chi aggiorna

- Le sessioni JSON salvate con la 0.1.0 restano leggibili: i nuovi campi hanno default.
  Rilanciando `aipaf summary` o `aipaf report` vengono ricalcolate con le nuove regole.
- `test_llm.py::test_empty_returns_empty_dict` è stato sostituito da test che verificano
  l'errore: se un fork del progetto dipendeva dal fallback `{}`, va adeguato.
- Il notebook `AIPAF_Assessment.ipynb` non è stato toccato: le celle che marcano i criteri
  non compilati come `score=0, L` continuano a funzionare ma non usano N/A né PARTIAL.

## [0.1.0] — 2026-04-28

Prima versione: Scoring Engine deterministico, LLM layer (Claude + Ollama), RAG layer
(ChromaDB, embedding Ollama/hash), report Markdown, CLI Typer con `assess`, `score`,
`summary`, `report`, `check`, `rag ingest`, `rag status`.
