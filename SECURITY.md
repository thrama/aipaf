# Security Policy

## Versioni supportate

Viene mantenuta solo l'ultima release pubblicata (vedi [CHANGELOG](CHANGELOG.md)).

## Segnalare una vulnerabilità

Non aprire una issue pubblica per problemi di sicurezza. Usa
[GitHub Security Advisories](../../security/advisories/new) ("Report a vulnerability")
per una segnalazione privata; riceverai un riscontro entro 7 giorni.

## Cosa NON è un problema di sicurezza di AIPAF

- Il contenuto dei report generati: AIPAF elabora le informazioni che l'utente inserisce e
  le invia al provider LLM configurato (Anthropic API o Ollama locale). La riservatezza di
  quei dati dipende dal provider scelto e dalla configurazione dell'utente.
- Chiavi API salvate in `.env`: il file è escluso dal versionamento; non committarlo.
