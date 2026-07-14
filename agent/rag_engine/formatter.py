from rag_engine.citation import citation_label


def format_context(results: list[dict]) -> str:
    if not results:
        return "未找到相關的索引內容，請先重建索引或加入來源文件。"

    contexts: list[str] = []

    for index, result in enumerate(results, start=1):
        doc = result["document"]
        meta = result["metadata"]
        rerank_score = result.get("rerank_score")
        labels = [
            f"[{citation_label(index)}]",
            f"來源：{meta.get('source_path', '')}",
            f"檔案：{meta.get('file_name', '')}",
            f"類型：{meta.get('file_type', '')}",
            f"檢索來源：{result.get('retrieval_source', 'vector')}",
            f"資料夾：{meta.get('folder_name', '')}",
            f"Chunk：{meta.get('chunk_id', '')}",
            f"切分策略：{meta.get('chunk_strategy', '')}",
            f"Rerank 分數：{rerank_score:.4f}" if rerank_score is not None else "Rerank 分數：N/A",
            f"BM25 分數：{result.get('bm25_score', 0.0):.4f}",
            f"關鍵字分數：{result.get('keyword_score', 0.0):.4f}",
            f"最終排序分數：{result.get('final_score', 0.0):.4f}",
        ]

        if "page_number" in meta:
            labels.append(f"頁碼：{meta['page_number']}")
        if "section_title" in meta:
            labels.append(f"章節：{meta['section_title']}")
        if "function_or_class" in meta:
            labels.append(f"函式/類別：{meta['function_or_class']}")

        contexts.append(f"{' | '.join(labels)}\n內容：{doc}")

    return "\n\n---\n\n".join(contexts)
