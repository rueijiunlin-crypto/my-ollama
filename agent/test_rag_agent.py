import os

os.environ["RAG_AGENT_SKIP_MODEL_LOAD"] = "1"

from rag_engine.chunker import split_markdown, split_python
from rag_engine.formatter import format_context
from rag_engine.retriever import keyword_score


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
        "build_index 做什麼",
        "這段程式會建立索引。",
        {
            "file_name": "rag_agent.py",
            "function_or_class": "build_index",
            "section_title": "",
        },
    )
    assert score > 0


def test_format_context() -> None:
    context = format_context(
        [
            {
                "document": "測試內容",
                "metadata": {
                    "source_path": r"D:\AI_Server\knowledge_base\test.md",
                    "file_name": "test.md",
                    "file_type": ".md",
                    "folder_name": "knowledge_base",
                    "chunk_id": 0,
                    "chunk_strategy": "markdown_heading",
                },
                "rerank_score": 0.8,
                "keyword_score": 0.5,
                "final_score": 0.875,
            }
        ]
    )
    assert "來源" in context
    assert "檔名" in context
    assert "Chunk策略" in context
    assert "Final分數" in context


def main() -> None:
    test_split_markdown()
    test_split_python()
    test_keyword_score()
    test_format_context()
    print("所有測試通過。")


if __name__ == "__main__":
    main()
