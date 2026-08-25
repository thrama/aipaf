"""
AIPAF — AI Project Assessment Framework
rag/retriever.py — Query interface per il vector store ChromaDB

Responsabilità:
  - Interroga le collection regulatory e company
  - Supporta query contestualizzate per criterio AIPAF
  - Formatta il contesto recuperato per l'iniezione nel prompt LLM
  - Deduplication dei chunk sovrapposti

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ingestor import DocumentIngestor


# ---------------------------------------------------------------------------
# Dataclass risultato
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Singolo chunk recuperato dal vector store."""
    text:        str
    filename:    str
    source_type: str   # "regulatory" | "company"
    version:     str
    chunk_index: int
    distance:    float  # distanza vettoriale (più bassa = più rilevante)

    def __str__(self) -> str:
        return (
            f"[{self.source_type.upper()} · {self.filename} · v{self.version}]\n"
            f"{self.text}"
        )


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Interfaccia di query sul vector store AIPAF.

    Usage:
        ingestor = DocumentIngestor(embedding_backend="ollama")
        retriever = RAGRetriever(ingestor)

        # Query generica su entrambe le collection
        chunks = retriever.query("obblighi trasparenza AI Act sistema HIGH RISK", n_results=4)

        # Query contestualizzata per un criterio specifico
        context = retriever.context_for_criterion("D2.C1", "Classificazione rischio", "...")

        # Solo documenti aziendali
        chunks = retriever.query("codice etico impatto occupazionale", sources=["company"])
    """

    def __init__(self, ingestor: DocumentIngestor):
        self._ingestor = ingestor

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results:  int = 4,
        sources:    Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Interroga il vector store e restituisce i chunk più rilevanti.

        Args:
            query_text: Testo della query in linguaggio naturale.
            n_results:  Numero massimo di chunk per collection interrogata.
            sources:    Lista di source type da interrogare.
                        Default: ["regulatory", "company"].
                        Passare ["regulatory"] per solo normativi,
                        ["company"] per solo aziendali.

        Returns:
            Lista di RetrievedChunk ordinata per rilevanza (distanza crescente).
        """
        if sources is None:
            sources = ["regulatory", "company"]

        all_chunks: list[RetrievedChunk] = []

        if "regulatory" in sources:
            chunks = self._query_collection(
                self._ingestor.col_regulatory,
                query_text,
                n_results,
                "regulatory",
            )
            all_chunks.extend(chunks)

        if "company" in sources:
            chunks = self._query_collection(
                self._ingestor.col_company,
                query_text,
                n_results,
                "company",
            )
            all_chunks.extend(chunks)

        # Ordina per distanza (rilevanza) e deduplicazione
        all_chunks.sort(key=lambda c: c.distance)
        return self._deduplicate(all_chunks)

    def context_for_criterion(
        self,
        criterion_id: str,
        criterion_name: str,
        criterion_description: str,
        source: Optional[str] = None,
        n_results: int = 3,
    ) -> str:
        """
        Recupera contesto normativo/aziendale pertinente a un criterio AIPAF
        e lo formatta per l'iniezione nel prompt LLM.

        Args:
            criterion_id:          Es. "D2.C1"
            criterion_name:        Es. "Classificazione rischio EU AI Act"
            criterion_description: Testo della descrizione del criterio
            source:                "regulatory" | "company" | None (entrambe)
            n_results:             Chunk per collection

        Returns:
            Stringa formattata pronta per il system prompt.
        """
        # Query composita: ID + nome + descrizione → migliore recall
        query = f"{criterion_id} {criterion_name} {criterion_description}"

        sources = [source] if source else None
        chunks = self.query(query, n_results=n_results, sources=sources)

        if not chunks:
            return ""

        lines = ["--- CONTESTO NORMATIVO E AZIENDALE RILEVANTE ---"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"\n[Fonte {i}: {chunk.source_type.upper()} · "
                f"{chunk.filename} · v{chunk.version}]\n"
                f"{chunk.text}"
            )
        lines.append("--- FINE CONTESTO ---")

        return "\n".join(lines)

    def list_indexed_documents(self) -> dict:
        """Restituisce il manifest dei documenti indicizzati."""
        return self._ingestor.status()

    # ------------------------------------------------------------------
    # Utility interne
    # ------------------------------------------------------------------

    def _query_collection(
        self,
        collection,
        query_text: str,
        n_results:  int,
        source_type: str,
    ) -> list[RetrievedChunk]:
        """Esegue la query su una singola collection."""
        if collection.count() == 0:
            return []

        # ChromaDB limita n_results al count della collection
        actual_n = min(n_results, collection.count())

        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=actual_n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        chunks: list[RetrievedChunk] = []
        docs      = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            if not doc:
                continue
            chunks.append(RetrievedChunk(
                text=doc,
                filename=meta.get("filename", ""),
                source_type=source_type,
                version=meta.get("version", ""),
                chunk_index=meta.get("chunk_index", 0),
                distance=dist,
            ))

        return chunks

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Rimuove chunk con testo quasi identico (overlap > 80%).
        Mantiene il chunk con distanza più bassa.
        """
        seen: list[str] = []
        deduped: list[RetrievedChunk] = []

        for chunk in chunks:
            if not any(self._overlap_ratio(chunk.text, s) > 0.8 for s in seen):
                deduped.append(chunk)
                seen.append(chunk.text)

        return deduped

    @staticmethod
    def _overlap_ratio(a: str, b: str) -> float:
        """Rapporto di overlap tra due stringhe (0.0 - 1.0)."""
        if not a or not b:
            return 0.0
        shorter = min(a, b, key=len)
        longer  = max(a, b, key=len)
        if shorter in longer:
            return len(shorter) / len(longer)
        # Token overlap approssimato
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union        = tokens_a | tokens_b
        return len(intersection) / len(union)
