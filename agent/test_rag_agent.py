import os
import tempfile
from pathlib import Path

os.environ["RAG_AGENT_SKIP_MODEL_LOAD"] = "1"

from rag_engine.chunker import split_markdown, split_python
from rag_engine.formatter import format_context
from rag_engine import retriever
from rag_engine.manifest import scan_source_files
from rag_engine.path_filter import is_excluded_path
from rag_engine.retriever import (
    bm25_search,
    keyword_score,
    normalize_scores,
    retrieve_docs,
    tokenize_for_bm25,
    tokenize_text,
)


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
    assert "Source:" in context
    assert "File:" in context
    assert "Retrieval:" in context
    assert "BM25 score:" in context
    assert "Final score:" in context


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
    print("All tests passed.")


if __name__ == "__main__":
    main()
