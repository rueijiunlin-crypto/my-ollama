import os
import tempfile
from pathlib import Path

os.environ["RAG_AGENT_SKIP_MODEL_LOAD"] = "1"

from rag_engine.chunker import (
    create_parent_child_chunks,
    split_markdown,
    split_python,
)
from rag_engine.citation import append_source_list, format_source_list
from rag_engine.formatter import format_context
from rag_engine import retriever
from rag_engine import file_reader, indexer, knowledge_manager
from rag_engine.manifest import scan_source_files
from rag_engine.path_filter import is_excluded_path
from rag_engine.retriever import (
    bm25_search,
    evaluate_retrieval_quality,
    invalidate_bm25_cache,
    metadata_matches,
    keyword_score,
    normalize_scores,
    retrieve_docs,
    select_diverse_results,
    tokenize_for_bm25,
    tokenize_text,
)
from llm import ollama_client
from main import format_main_help, format_qa_help


def test_split_markdown() -> None:
    text = """# Transformer
Transformer intro.

## Self Attention
Self Attention content.
"""
    chunks = split_markdown(text)
    assert len(chunks) >= 2
    assert any("Transformer" in chunk for chunk in chunks)
    assert any("Self Attention" in chunk for chunk in chunks)


def test_split_python() -> None:
    text = '''
class Planner:
    def run(self):
        return "ok"


def build_index():
    return True
'''
    chunks = split_python(text)
    assert len(chunks) >= 2
    assert any("class Planner" in chunk for chunk in chunks)
    assert any("def build_index" in chunk for chunk in chunks)


def test_keyword_score() -> None:
    score = keyword_score(
        "build_index是什麼",
        "This document explains the search pipeline.",
        {
            "file_name": "rag_agent.py",
            "function_or_class": "build_index",
            "section_title": "",
        },
    )
    assert score > 0


def test_mixed_language_tokenizer() -> None:
    token_query = tokenize_text("token是啥")
    assert "token" in token_query
    assert "是啥" in token_query

    function_query = tokenize_text("build_index是什麼")
    assert "build_index" in function_query
    assert "build" in function_query
    assert "index" in function_query
    assert "是什" in function_query
    assert "什麼" in function_query


def test_tokenize_for_bm25() -> None:
    tokens = tokenize_for_bm25("/camera/color/image_raw rclcpp::Node build_index")
    assert "camera" in tokens
    assert "color" in tokens
    assert "image" in tokens
    assert "raw" in tokens
    assert "rclcpp" in tokens
    assert "node" in tokens
    assert "build" in tokens
    assert "index" in tokens


def test_normalize_scores() -> None:
    scores = normalize_scores([10, 20, 30])
    assert scores == [0.0, 0.5, 1.0]
    assert normalize_scores([]) == []
    assert normalize_scores([5, 5]) == [1.0, 1.0]


def test_excluded_paths() -> None:
    assert is_excluded_path(Path("project/.venv/Lib/site-packages/example.py"))
    assert is_excluded_path(Path("project/.git/config"))
    assert is_excluded_path(Path("project/chroma_db/chroma.sqlite3"))
    assert not is_excluded_path(Path("project/knowledge_base/note.md"))


def test_scan_source_files_excludes_directories() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        included = root / "knowledge_base" / "note.md"
        excluded_venv = root / ".venv" / "Lib" / "site-packages" / "package.py"
        excluded_git = root / ".git" / "hook.py"
        excluded_chroma = root / "chroma_db" / "internal.py"

        for path in (included, excluded_venv, excluded_git, excluded_chroma):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test", encoding="utf-8")

        results = scan_source_files([root])

        assert results == [included]
        assert all(".venv" not in str(path) for path in results)
        assert all("site-packages" not in str(path) for path in results)


def test_bm25_search_with_fake_documents() -> None:
    documents = [
        {
            "id": "chunk-1",
            "document": "The build_index function creates ChromaDB chunks.",
            "metadata": {"source_path": "a.md", "chunk_id": "1"},
        },
        {
            "id": "chunk-2",
            "document": "A cooking note about soup.",
            "metadata": {"source_path": "b.md", "chunk_id": "2"},
        },
    ]
    results = bm25_search("build_index chunks", top_k=1, indexed_documents=documents)
    assert len(results) == 1
    assert results[0]["id"] == "chunk-1"
    assert results[0]["bm25_score"] > 0


class FakeCollection:
    def count(self) -> int:
        return 2

    def query(self, **kwargs):
        return {
            "ids": [["chunk-1"]],
            "documents": [["The build_index function creates ChromaDB chunks."]],
            "metadatas": [[{"source_path": "a.md", "chunk_id": "1", "file_name": "a.md"}]],
            "distances": [[0.1]],
        }

    def get(self, include=None):
        return {
            "ids": ["chunk-1", "chunk-2"],
            "documents": [
                "The build_index function creates ChromaDB chunks.",
                "A cooking note about soup.",
            ],
            "metadatas": [
                {"source_path": "a.md", "chunk_id": "1", "file_name": "a.md"},
                {"source_path": "b.md", "chunk_id": "2", "file_name": "b.md"},
            ],
        }


def test_retrieve_docs_includes_bm25_fields() -> None:
    original_get_collection = retriever.get_collection
    original_encode_text = retriever.encode_text
    original_get_reranker_model = retriever.get_reranker_model

    retriever.get_collection = lambda: FakeCollection()
    retriever.encode_text = lambda text: [0.1, 0.2]
    retriever.get_reranker_model = lambda: None

    try:
        results = retrieve_docs("build_index chunks")
    finally:
        retriever.get_collection = original_get_collection
        retriever.encode_text = original_encode_text
        retriever.get_reranker_model = original_get_reranker_model

    assert results
    assert "document" in results[0]
    assert "metadata" in results[0]
    assert "keyword_score" in results[0]
    assert "final_score" in results[0]
    assert "bm25_score" in results[0]
    assert "retrieval_source" in results[0]


class FakeReranker:
    def predict(self, pairs):
        return [0.5 for _ in pairs]


def test_reranker_final_score_uses_keyword_only_as_small_bonus() -> None:
    original_get_collection = retriever.get_collection
    original_encode_text = retriever.encode_text
    original_get_reranker_model = retriever.get_reranker_model

    retriever.get_collection = lambda: FakeCollection()
    retriever.encode_text = lambda text: [0.1, 0.2]
    retriever.get_reranker_model = lambda: FakeReranker()

    try:
        results = retrieve_docs("build_index是什麼")
    finally:
        retriever.get_collection = original_get_collection
        retriever.encode_text = original_encode_text
        retriever.get_reranker_model = original_get_reranker_model

    assert results
    top_result = results[0]
    expected = 0.5 + retriever.KEYWORD_WEIGHT * top_result["keyword_score"]
    assert abs(top_result["final_score"] - expected) < 1e-9
    assert top_result["bm25_score"] > 0


def test_format_context() -> None:
    context = format_context(
        [
            {
                "document": "Search context",
                "metadata": {
                    "source_path": r"G:\AI_Server\knowledge_base\test.md",
                    "file_name": "test.md",
                    "file_type": ".md",
                    "folder_name": "knowledge_base",
                    "chunk_id": 0,
                    "chunk_strategy": "markdown_heading",
                },
                "rerank_score": 0.8,
                "keyword_score": 0.5,
                "bm25_score": 0.3,
                "final_score": 0.875,
                "retrieval_source": "vector,bm25",
            }
        ]
    )
    assert "[來源 1]" in context
    assert "來源：" in context
    assert "檔案：" in context
    assert "檢索來源：" in context
    assert "BM25 分數：" in context
    assert "最終排序分數：" in context


def test_citation_source_list() -> None:
    results = [
        {
            "document": "Token 是模型處理文字的基本單位。",
            "metadata": {
                "source_path": r"G:\AI_Server\knowledge_base\notes.md",
                "file_name": "notes.md",
                "section_title": "Token",
                "page_number": 3,
            },
        }
    ]
    sources = format_source_list(results)
    assert "[來源 1]" in sources
    assert "notes.md" in sources
    assert "章節：Token" in sources
    assert "頁碼：3" in sources

    answer = append_source_list("Token 是基本單位。[來源 1]", results)
    assert answer.count("參考來源：") == 1
    assert "---" in answer


def test_retrieval_quality_accepts_semantic_evidence() -> None:
    quality = evaluate_retrieval_quality(
        [
            {
                "rerank_score": 0.95,
                "bm25_score": 0.0,
                "keyword_score": 0.0,
                "final_score": 0.95,
            }
        ]
    )
    assert quality["accepted"] is True
    assert quality["relevant_count"] == 1


def test_retrieval_quality_rejects_weak_evidence() -> None:
    quality = evaluate_retrieval_quality(
        [
            {
                "rerank_score": 0.0001,
                "bm25_score": 0.2,
                "keyword_score": 0.0,
                "final_score": 0.0001,
            }
        ]
    )
    assert quality["accepted"] is False
    assert "沒有找到足夠可靠" in quality["reason"]


def test_ollama_health_check_finds_configured_model() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": ollama_client.MODEL}]}

    class FakeRequests:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(url: str, timeout: float):
            assert url.endswith("/api/tags")
            assert timeout > 0
            return FakeResponse()

    original_requests = ollama_client.requests
    ollama_client.requests = FakeRequests
    try:
        health = ollama_client.check_ollama_health()
    finally:
        ollama_client.requests = original_requests

    assert health["available"] is True
    assert health["model_available"] is True
    assert ollama_client.MODEL in health["message"]


def test_query_rewrite_uses_recent_context() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": "Token 與 Embedding 有什麼差別？"}

    class FakeRequests:
        class RequestException(Exception):
            pass

        @staticmethod
        def post(url: str, json: dict, timeout: float):
            assert "Token 是什麼" in json["prompt"]
            assert timeout > 0
            return FakeResponse()

    original_requests = ollama_client.requests
    ollama_client.requests = FakeRequests
    try:
        rewritten = ollama_client.rewrite_query(
            "那它跟 Embedding 有什麼差別？",
            [{"question": "Token 是什麼？", "answer": "Token 是文字單位。"}],
        )
    finally:
        ollama_client.requests = original_requests

    assert rewritten == "Token 與 Embedding 有什麼差別？"


def test_metadata_filter() -> None:
    metadata = {
        "week": "Week03",
        "file_type": ".py",
        "root_source": "learning",
    }
    assert metadata_matches(metadata, {"week": "week03"})
    assert metadata_matches(metadata, {"file_type": "py"})
    assert not metadata_matches(metadata, {"week": "Week01"})
    assert metadata_matches(metadata, {"unsupported": "ignored"})


def test_parent_child_chunking() -> None:
    text = "# Transformer\n\n" + ("Self Attention 會計算權重。" * 100)
    records = create_parent_child_chunks(text, ".md", "abc123")
    assert len(records) >= 2
    assert all(record["parent_id"].startswith("abc123_p") for record in records)
    assert all(record["child_text"] in record["parent_text"] for record in records)
    assert len({record["parent_id"] for record in records}) >= 1


def test_context_diversity_and_parent_expansion() -> None:
    results = [
        {
            "document": "child one",
            "metadata": {
                "source_path": "a.md",
                "parent_id": "p1",
                "parent_text": "parent one complete context",
            },
            "final_score": 1.0,
        },
        {
            "document": "child two",
            "metadata": {
                "source_path": "a.md",
                "parent_id": "p1",
                "parent_text": "parent one complete context",
            },
            "final_score": 0.9,
        },
        {
            "document": "different child",
            "metadata": {
                "source_path": "b.md",
                "parent_id": "p2",
                "parent_text": "different parent context",
            },
            "final_score": 0.8,
        },
    ]
    selected = select_diverse_results(results)
    assert len(selected) == 2
    assert selected[0]["document"] == "parent one complete context"
    assert selected[0]["matched_child"] == "child one"
    assert "parent_text" not in selected[0]["metadata"]


def test_bm25_cache_avoids_reloading_documents() -> None:
    class CountingCollection(FakeCollection):
        def __init__(self):
            self.get_calls = 0

        def get(self, include=None):
            self.get_calls += 1
            return super().get(include=include)

    collection = CountingCollection()
    original_get_collection = retriever.get_collection
    retriever.get_collection = lambda: collection
    invalidate_bm25_cache()
    try:
        bm25_search("build_index")
        bm25_search("chunks")
    finally:
        retriever.get_collection = original_get_collection
        invalidate_bm25_cache()

    assert collection.get_calls == 1


def test_batch_add_rolls_back_partial_write() -> None:
    class FailingCollection:
        def __init__(self):
            self.calls = 0

        def add(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated failure")

    records = [
        {"id": f"id-{index}", "document": f"doc-{index}", "metadata": {"i": index}}
        for index in range(65)
    ]
    collection = FailingCollection()
    deleted: list[str] = []
    original_get_collection = indexer.get_collection
    original_encode_texts = indexer.encode_texts
    original_delete = indexer.delete_file_chunks
    indexer.get_collection = lambda: collection
    indexer.encode_texts = lambda texts, batch_size: [[0.1] for _ in texts]
    indexer.delete_file_chunks = lambda ids: deleted.extend(ids) or True
    try:
        success = indexer._add_records(records)
    finally:
        indexer.get_collection = original_get_collection
        indexer.encode_texts = original_encode_texts
        indexer.delete_file_chunks = original_delete

    assert success is False
    assert len(deleted) == 64


def test_parent_child_ids_are_unique_across_pdf_pages() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "paper.pdf"
        path.write_bytes(b"fake pdf")
        docs = [
            {
                "path": path,
                "text": "第一頁內容 " * 100,
                "metadata": {"file_type": ".pdf", "page_number": 1},
            },
            {
                "path": path,
                "text": "第二頁內容 " * 100,
                "metadata": {"file_type": ".pdf", "page_number": 2},
            },
        ]
        original_read = indexer.read_single_file
        indexer.read_single_file = lambda source_path: docs
        try:
            prepared = indexer._prepare_single_file(path, "a" * 64)
        finally:
            indexer.read_single_file = original_read

    assert prepared is not None
    _, records = prepared
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
    assert any("_d0_" in item for item in ids)
    assert any("_d1_" in item for item in ids)


def test_incremental_update_adds_new_before_deleting_old() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "notes.md"
        path.write_text("new content", encoding="utf-8")
        manifest_path = Path(temporary_directory) / "index_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        manifest = {
            "embedding_model": indexer.EMBEDDING_MODEL,
            "collection_name": indexer.COLLECTION_NAME,
            "index_schema_version": indexer.INDEX_SCHEMA_VERSION,
            "data_dirs": [],
            "files": {
                str(path): {
                    "sha256": "old",
                    "chunk_ids": ["old-id"],
                }
            },
        }
        events: list[str] = []

        originals = {
            "manifest_path": indexer.MANIFEST_PATH,
            "load_manifest": indexer.load_manifest,
            "save_manifest": indexer.save_manifest,
            "scan_source_files": indexer.scan_source_files,
            "calculate_file_sha256": indexer.calculate_file_sha256,
            "index_single_file": indexer.index_single_file,
            "delete_file_chunks": indexer.delete_file_chunks,
            "invalidate": indexer._invalidate_retrieval_cache,
        }
        indexer.MANIFEST_PATH = manifest_path
        indexer.load_manifest = lambda: manifest
        indexer.save_manifest = lambda value: events.append("save")
        indexer.scan_source_files = lambda folders: [path]
        indexer.calculate_file_sha256 = lambda source_path: "new"
        indexer.index_single_file = lambda source_path, sha: (
            events.append("add_new")
            or {"sha256": sha, "chunk_ids": ["new-id"], "parent_count": 1}
        )
        indexer.delete_file_chunks = lambda ids: events.append(f"delete:{ids[0]}") or True
        indexer._invalidate_retrieval_cache = lambda: None
        try:
            success = indexer.build_index(full_rebuild=False)
        finally:
            indexer.MANIFEST_PATH = originals["manifest_path"]
            indexer.load_manifest = originals["load_manifest"]
            indexer.save_manifest = originals["save_manifest"]
            indexer.scan_source_files = originals["scan_source_files"]
            indexer.calculate_file_sha256 = originals["calculate_file_sha256"]
            indexer.index_single_file = originals["index_single_file"]
            indexer.delete_file_chunks = originals["delete_file_chunks"]
            indexer._invalidate_retrieval_cache = originals["invalidate"]

    assert success is True
    assert events.index("add_new") < events.index("delete:old-id")
    assert events[-1] == "save"


def test_full_rebuild_stops_when_clear_fails() -> None:
    original_clear = indexer.clear_collection
    indexer.clear_collection = lambda: False
    try:
        assert indexer.build_index(full_rebuild=True) is False
    finally:
        indexer.clear_collection = original_clear


def test_knowledge_manager_status() -> None:
    manifest = {
        "embedding_model": "BAAI/bge-m3",
        "collection_name": "test_collection",
        "data_dirs": ["knowledge_base"],
        "files": {
            "knowledge_base/notes.md": {
                "indexed_at": "2026-07-14 10:00:00",
                "parent_count": 2,
            }
        },
    }

    class CountCollection:
        def count(self) -> int:
            return 4

    original_load = knowledge_manager.load_manifest
    original_collection = knowledge_manager.get_collection
    knowledge_manager.load_manifest = lambda: manifest
    knowledge_manager.get_collection = lambda: CountCollection()
    try:
        status = knowledge_manager.get_knowledge_status()
        output = knowledge_manager.format_knowledge_status(status)
    finally:
        knowledge_manager.load_manifest = original_load
        knowledge_manager.get_collection = original_collection

    assert status["file_count"] == 1
    assert status["chunk_count"] == 4
    assert status["parent_count"] == 2
    assert "索引文件數：1" in output


def test_pdf_extraction_report_metadata() -> None:
    class FakePage:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, path: str):
            self.pages = [FakePage("有效內容"), FakePage("")]

    original_reader = file_reader.PdfReader
    file_reader.PdfReader = FakeReader
    try:
        docs = file_reader._read_pdf_pages(
            Path("paper.pdf"),
            {"file_name": "paper.pdf", "file_type": ".pdf"},
        )
    finally:
        file_reader.PdfReader = original_reader

    assert len(docs) == 1
    assert docs[0]["metadata"]["page_number"] == 1
    assert docs[0]["metadata"]["pdf_total_pages"] == 2


def test_help_guides() -> None:
    main_help = format_main_help()
    qa_help = format_qa_help()
    manager_help = knowledge_manager.format_knowledge_manager_help()

    assert "help" in main_help
    assert "q" in main_help
    assert "filter show" in qa_help
    assert "filter week Week03" in qa_help
    assert "clip" in manager_help
    assert "Week03" in manager_help


def main() -> None:
    test_split_markdown()
    test_split_python()
    test_keyword_score()
    test_mixed_language_tokenizer()
    test_tokenize_for_bm25()
    test_normalize_scores()
    test_excluded_paths()
    test_scan_source_files_excludes_directories()
    test_bm25_search_with_fake_documents()
    test_retrieve_docs_includes_bm25_fields()
    test_reranker_final_score_uses_keyword_only_as_small_bonus()
    test_format_context()
    test_citation_source_list()
    test_retrieval_quality_accepts_semantic_evidence()
    test_retrieval_quality_rejects_weak_evidence()
    test_ollama_health_check_finds_configured_model()
    test_query_rewrite_uses_recent_context()
    test_metadata_filter()
    test_parent_child_chunking()
    test_context_diversity_and_parent_expansion()
    test_bm25_cache_avoids_reloading_documents()
    test_batch_add_rolls_back_partial_write()
    test_parent_child_ids_are_unique_across_pdf_pages()
    test_incremental_update_adds_new_before_deleting_old()
    test_full_rebuild_stops_when_clear_fails()
    test_knowledge_manager_status()
    test_pdf_extraction_report_metadata()
    test_help_guides()
    print("All tests passed.")


if __name__ == "__main__":
    main()
