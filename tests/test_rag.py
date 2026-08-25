"""
AIPAF — Test suite per rag/ingestor.py e rag/retriever.py

Usa HashEmbeddingFunction (zero dipendenze esterne, zero download modelli)
e PersistentClient ChromaDB su directory temporanea.

Eseguire: python -m pytest tests/test_rag.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from aipaf.rag.ingestor import (
    DocumentIngestor,
    DocumentSource,
    HashEmbeddingFunction,
    chunk_text,
)
from aipaf.rag.retriever import RAGRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path):
    (tmp_path / "docs" / "sources" / "regulatory").mkdir(parents=True)
    (tmp_path / "docs" / "sources" / "company").mkdir(parents=True)
    (tmp_path / "docs" / "index").mkdir(parents=True)
    (tmp_path / "docs" / "manifest.json").write_text(
        '{"regulatory": {}, "company": {}}', encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def sample_regulatory(tmp_project: Path) -> Path:
    content = (
        "EU AI Act — Articolo 14: Supervisione umana\n\n"
        "I sistemi di IA ad alto rischio devono essere progettati e sviluppati "
        "in modo tale da consentire una supervisione umana efficace. "
        "I fornitori devono garantire che il sistema possa essere monitorato "
        "da persone fisiche durante il periodo in cui è in uso.\n\n"
        "Articolo 13: Trasparenza e fornitura di informazioni\n\n"
        "I sistemi di IA ad alto rischio devono essere progettati in modo da "
        "garantire che il loro funzionamento sia sufficientemente trasparente "
        "da consentire agli utilizzatori di interpretare correttamente l'output."
    )
    path = tmp_project / "docs" / "sources" / "regulatory" / "eu_ai_act_test.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_company(tmp_project: Path) -> Path:
    content = (
        "Codice Etico Acme S.p.A. — Sezione 7: AI Responsabile\n\n"
        "L'azienda si impegna a sviluppare e utilizzare sistemi di intelligenza "
        "artificiale in modo responsabile, trasparente e rispettoso dei diritti "
        "delle persone. Ogni progetto AI deve essere valutato per l'impatto "
        "occupazionale e prevedere piani di reskilling per i dipendenti coinvolti.\n\n"
        "Principio di Non Discriminazione: i sistemi AI non devono perpetuare "
        "bias o discriminazioni basate su caratteristiche protette."
    )
    path = tmp_project / "docs" / "sources" / "company" / "codice_etico_test.txt"
    path.write_text(content, encoding="utf-8")
    return path


def make_ingestor(tmp_project: Path) -> DocumentIngestor:
    index_path    = tmp_project / "docs" / "index"
    manifest_path = tmp_project / "docs" / "manifest.json"
    with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
         patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path), \
         patch("aipaf.rag.ingestor.REGULATORY_DIR", tmp_project / "docs" / "sources" / "regulatory"), \
         patch("aipaf.rag.ingestor.COMPANY_DIR",    tmp_project / "docs" / "sources" / "company"):
        return DocumentIngestor(index_path=index_path, embedding_backend="hash")


# ---------------------------------------------------------------------------
# HashEmbeddingFunction
# ---------------------------------------------------------------------------

class TestHashEmbeddingFunction:

    def test_output_shape(self):
        ef = HashEmbeddingFunction()
        result = ef(["testo di prova", "altro testo"])
        assert len(result) == 2
        assert len(result[0]) == HashEmbeddingFunction.DIM

    def test_deterministic(self):
        ef = HashEmbeddingFunction()
        a, b = ef(["stesso testo"]), ef(["stesso testo"])
        assert [list(map(float, v)) for v in a] == [list(map(float, v)) for v in b]

    def test_chromadb_interface(self):
        """ChromaDB 1.5.x: senza name()/embed_query le query falliscono (bug v0.2.0)."""
        from chromadb.api.types import EmbeddingFunction
        ef = HashEmbeddingFunction()
        assert isinstance(ef, EmbeddingFunction)
        assert HashEmbeddingFunction.name() == "hash_embedding_function"
        assert hasattr(ef, "embed_query")
        assert len(ef.embed_query(["q"])[0]) == HashEmbeddingFunction.DIM

    def test_different_inputs_different_outputs(self):
        ef = HashEmbeddingFunction()
        a, b = ef(["testo A"]), ef(["testo B"])
        assert list(map(float, a[0])) != list(map(float, b[0]))

    def test_values_in_range(self):
        ef = HashEmbeddingFunction()
        for v in ef(["test"])[0]:
            assert -1.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Testo breve.", chunk_size=800)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        chunks = chunk_text("Paragrafo. " * 100, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunks_not_empty(self):
        chunks = chunk_text("Frase. " * 200, chunk_size=200, chunk_overlap=40)
        assert all(c.strip() for c in chunks)

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []


# ---------------------------------------------------------------------------
# DocumentIngestor
# ---------------------------------------------------------------------------

class TestDocumentIngestor:

    def test_ingest_regulatory_file(self, tmp_project, sample_regulatory):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            doc = ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
        assert doc is not None
        assert doc.source_type == DocumentSource.REGULATORY
        assert doc.chunk_count > 0

    def test_ingest_company_file(self, tmp_project, sample_company):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            doc = ingestor.ingest_file(sample_company, DocumentSource.COMPANY)
        assert doc is not None
        assert doc.chunk_count > 0

    def test_skip_already_ingested(self, tmp_project, sample_regulatory):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            doc1 = ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            doc2 = ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
        assert doc1 is not None
        assert doc2 is None  # saltato: checksum identico

    def test_reingest_when_index_lost_but_manifest_current(self, tmp_project, sample_regulatory):
        """Manifest aggiornato ma chunk assenti dall'indice (es. docs/index/ cancellato,
        o manifest proveniente da un altro ambiente): l'ingestor deve re-indicizzare,
        non fidarsi del solo checksum."""
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            doc1 = ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            assert doc1 is not None
            # Simula la perdita dell'indice lasciando intatto il manifest
            ingestor._delete_file_chunks(ingestor.col_regulatory, sample_regulatory.name)
            assert ingestor.col_regulatory.count() == 0
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert sample_regulatory.name in manifest["regulatory"]

            doc2 = ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
        assert doc2 is not None
        assert doc2.chunk_count == doc1.chunk_count
        assert ingestor.col_regulatory.count() == doc1.chunk_count

    def test_force_reingest(self, tmp_project, sample_regulatory):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            doc2 = ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY, force=True)
        assert doc2 is not None

    def test_manifest_updated(self, tmp_project, sample_regulatory):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["regulatory"].get("eu_ai_act_test.txt")
        assert entry is not None
        assert "checksum" in entry
        assert "ingested_at" in entry
        assert entry["chunk_count"] > 0

    def test_status_counts(self, tmp_project, sample_regulatory, sample_company):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            ingestor.ingest_file(sample_company,    DocumentSource.COMPANY)
            status = ingestor.status()
        assert status["regulatory"]["documents"] == 1
        assert status["regulatory"]["chunks"] > 0
        assert status["company"]["documents"] == 1
        assert status["company"]["chunks"] > 0

    def test_delete_file(self, tmp_project, sample_regulatory):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            deleted = ingestor.delete_file("eu_ai_act_test.txt", DocumentSource.REGULATORY)
            status  = ingestor.status()
        assert deleted is True
        assert status["regulatory"]["documents"] == 0
        assert status["regulatory"]["chunks"] == 0


class TestIndexReopen:

    def _paths(self, tmp_project):
        return tmp_project / "docs" / "index", tmp_project / "docs" / "manifest.json"

    def test_reopen_without_backend_uses_persisted(self, tmp_project, sample_regulatory):
        index_path, manifest_path = self._paths(tmp_project)
        with patch("aipaf.rag.ingestor._INDEX_PATH", index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            a = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            a.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            del a
            b = DocumentIngestor(index_path=index_path, embedding_backend=None)
            assert b.status()["regulatory"]["chunks"] > 0
            assert RAGRetriever(b).query("supervisione", n_results=1)
            assert DocumentIngestor.persisted_backend(index_path) == "hash"

    def test_none_on_empty_index_raises(self, tmp_project):
        index_path, manifest_path = self._paths(tmp_project)
        with patch("aipaf.rag.ingestor._INDEX_PATH", index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            with pytest.raises(RuntimeError, match="non inizializzato"):
                DocumentIngestor(index_path=index_path, embedding_backend=None)
        assert DocumentIngestor.persisted_backend(index_path) is None

    def test_backend_conflict_is_explained(self, tmp_project):
        """Indice creato con 'hash', riaperto con 'ollama': messaggio con rimedio, non stack ChromaDB."""
        index_path, manifest_path = self._paths(tmp_project)
        with patch("aipaf.rag.ingestor._INDEX_PATH", index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            DocumentIngestor(index_path=index_path, embedding_backend="hash")
            with pytest.raises(RuntimeError, match="creato con embedding 'hash'"):
                DocumentIngestor(index_path=index_path, embedding_backend="ollama")


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class TestRAGRetriever:

    def _make_retriever(self, tmp_project, sample_regulatory, sample_company):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            ingestor.ingest_file(sample_regulatory, DocumentSource.REGULATORY)
            ingestor.ingest_file(sample_company,    DocumentSource.COMPANY)
            return RAGRetriever(ingestor)

    def test_query_regulatory_only(self, tmp_project, sample_regulatory, sample_company):
        retriever = self._make_retriever(tmp_project, sample_regulatory, sample_company)
        results = retriever.query("trasparenza", n_results=2, sources=["regulatory"])
        assert all(r.source_type == "regulatory" for r in results)

    def test_query_company_only(self, tmp_project, sample_regulatory, sample_company):
        retriever = self._make_retriever(tmp_project, sample_regulatory, sample_company)
        results = retriever.query("codice etico", n_results=2, sources=["company"])
        assert all(r.source_type == "company" for r in results)

    def test_context_for_criterion_returns_string(self, tmp_project, sample_regulatory, sample_company):
        retriever = self._make_retriever(tmp_project, sample_regulatory, sample_company)
        context = retriever.context_for_criterion(
            criterion_id="D2.C1",
            criterion_name="Classificazione rischio EU AI Act",
            criterion_description="Classificazione corretta nei 4 livelli di rischio",
        )
        assert isinstance(context, str)
        assert context != ""                       # v0.2.0: tornava "" per AttributeError silenziosa
        assert "CONTESTO NORMATIVO" in context
        assert "eu_ai_act_test.txt" in context

    def test_query_returns_chunks(self, tmp_project, sample_regulatory, sample_company):
        """Con collection popolate il retriever DEVE restituire chunk (nearest neighbour)."""
        retriever = self._make_retriever(tmp_project, sample_regulatory, sample_company)
        results = retriever.query("supervisione umana", n_results=2)
        assert len(results) >= 1
        assert all(r.distance >= 0 for r in results)

    def test_context_empty_collection(self, tmp_project):
        index_path    = tmp_project / "docs" / "index"
        manifest_path = tmp_project / "docs" / "manifest.json"
        with patch("aipaf.rag.ingestor._INDEX_PATH",    index_path), \
             patch("aipaf.rag.ingestor._MANIFEST_PATH", manifest_path):
            ingestor  = DocumentIngestor(index_path=index_path, embedding_backend="hash")
            retriever = RAGRetriever(ingestor)
        assert retriever.context_for_criterion("D1.C1", "Test", "Desc") == ""

    def test_list_indexed_documents(self, tmp_project, sample_regulatory, sample_company):
        retriever = self._make_retriever(tmp_project, sample_regulatory, sample_company)
        docs = retriever.list_indexed_documents()
        assert docs["regulatory"]["documents"] == 1
        assert docs["company"]["documents"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
