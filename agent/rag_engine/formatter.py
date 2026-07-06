def format_context(results: list[dict]) -> str:
    if not results:
        return "目前沒有可用的索引內容，請先建立或重建索引。"

    contexts: list[str] = []

    for result in results:
        doc = result["document"]
        meta = result["metadata"]
        rerank_score = result.get("rerank_score")
        labels = [
            f"來源：{meta.get('source_path', '')}",
            f"檔名：{meta.get('file_name', '')}",
            f"類型：{meta.get('file_type', '')}",
            f"資料夾：{meta.get('folder_name', '')}",
            f"chunk：{meta.get('chunk_id', '')}",
            f"Chunk策略：{meta.get('chunk_strategy', '')}",
            f"Rerank分數：{rerank_score:.4f}" if rerank_score is not None else "Rerank分數：N/A",
            f"Keyword分數：{result.get('keyword_score', 0.0):.4f}",
            f"Final分數：{result.get('final_score', 0.0):.4f}",
        ]

        if "page_number" in meta:
            labels.append(f"頁碼：{meta['page_number']}")
        if "section_title" in meta:
            labels.append(f"章節：{meta['section_title']}")
        if "function_or_class" in meta:
            labels.append(f"函式/類別：{meta['function_or_class']}")

        contexts.append(f"{' | '.join(labels)}\n內容：{doc}")

    return "\n\n---\n\n".join(contexts)
