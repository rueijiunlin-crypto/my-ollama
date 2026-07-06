import re

from config import RERANK_TOP_K, VECTOR_SEARCH_TOP_K
from rag_engine.formatter import format_context
from rag_engine.models import encode_text, get_collection, get_reranker_model


def keyword_score(question: str, document: str, metadata: dict) -> float:
    tokens = [
        token.lower()
        for token in re.split(r"[\s/\\_:.\-()[\]{}<>#，。；：、,;]+", question)
        if len(token.strip()) >= 2
    ]
    if not tokens:
        return 0.0

    document_lower = document.lower()
    file_name = str(metadata.get("file_name", "")).lower()
    function_or_class = str(metadata.get("function_or_class", "")).lower()
    section_title = str(metadata.get("section_title", "")).lower()

    score = 0.0
    max_score = len(tokens) * 4.5

    for token in tokens:
        if token in document_lower:
            score += 1.0
        if token in file_name:
            score += 1.5
        if token in function_or_class:
            score += 2.0
        if token in section_title:
            score += 2.0

    return min(score / max_score, 1.0)


def _vector_rank_score(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return max(0.0, 1.0 - (rank / (total - 1)))


def retrieve_docs(question: str) -> list[dict]:
    collection = get_collection()
    total_docs = collection.count()
    if total_docs == 0:
        return []

    query_embedding = encode_text(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(VECTOR_SEARCH_TOP_K, total_docs),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    candidates: list[dict] = []

    for rank, (document, metadata) in enumerate(zip(documents, metadatas)):
        kw_score = keyword_score(question, document, metadata)
        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "rerank_score": None,
                "keyword_score": kw_score,
                "vector_rank_score": _vector_rank_score(rank, len(documents)),
                "final_score": 0.0,
            }
        )

    if not candidates:
        return []

    reranker_model = get_reranker_model()
    if reranker_model is not None:
        pairs = [[question, candidate["document"]] for candidate in candidates]
        try:
            scores = reranker_model.predict(pairs)
            for candidate, score in zip(candidates, scores):
                rerank_score = float(score)
                candidate["rerank_score"] = rerank_score
                candidate["final_score"] = rerank_score + 0.15 * candidate["keyword_score"]
        except Exception as exc:
            print(f"Reranker 評分失敗，改用向量搜尋結果：{exc}")
            for candidate in candidates:
                candidate["final_score"] = (
                    candidate["vector_rank_score"] + 0.25 * candidate["keyword_score"]
                )
    else:
        for candidate in candidates:
            candidate["final_score"] = (
                candidate["vector_rank_score"] + 0.25 * candidate["keyword_score"]
            )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    return candidates[:RERANK_TOP_K]


def search_docs(question: str) -> str:
    return format_context(retrieve_docs(question))
