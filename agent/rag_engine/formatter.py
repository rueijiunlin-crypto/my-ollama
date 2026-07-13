def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant indexed context was found. Please rebuild the index or add source files."

    contexts: list[str] = []

    for result in results:
        doc = result["document"]
        meta = result["metadata"]
        rerank_score = result.get("rerank_score")
        labels = [
            f"Source: {meta.get('source_path', '')}",
            f"File: {meta.get('file_name', '')}",
            f"Type: {meta.get('file_type', '')}",
            f"Retrieval: {result.get('retrieval_source', 'vector')}",
            f"Folder: {meta.get('folder_name', '')}",
            f"Chunk: {meta.get('chunk_id', '')}",
            f"Chunk strategy: {meta.get('chunk_strategy', '')}",
            f"Rerank score: {rerank_score:.4f}" if rerank_score is not None else "Rerank score: N/A",
            f"BM25 score: {result.get('bm25_score', 0.0):.4f}",
            f"Keyword score: {result.get('keyword_score', 0.0):.4f}",
            f"Final score: {result.get('final_score', 0.0):.4f}",
        ]

        if "page_number" in meta:
            labels.append(f"Page: {meta['page_number']}")
        if "section_title" in meta:
            labels.append(f"Section: {meta['section_title']}")
        if "function_or_class" in meta:
            labels.append(f"Function/Class: {meta['function_or_class']}")

        contexts.append(f"{' | '.join(labels)}\nContent: {doc}")

    return "\n\n---\n\n".join(contexts)
