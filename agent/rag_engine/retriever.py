import re

from config import (
    BM25_ENABLED,
    BM25_TOP_K,
    BM25_WEIGHT,
    KEYWORD_WEIGHT,
    RERANK_TOP_K,
    RERANK_WEIGHT,
    VECTOR_SEARCH_TOP_K,
)
from rag_engine.formatter import format_context
from rag_engine.models import encode_text, get_collection, get_reranker_model

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:
    BM25Okapi = None


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    normalized = normalized.replace("::", " ")
    tokens = re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", normalized.lower())
    return [token for token in tokens if len(token) >= 2]


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 if value > 0 else 0.0 for value in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def keyword_score(question: str, document: str, metadata: dict) -> float:
    tokens = [
        token.lower()
        for token in re.split(r"[\s/\\_:.\-()[\]{}<>#嚗?嚗?;]+", question)
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


def _candidate_key(candidate_id: str | None, metadata: dict) -> str:
    if candidate_id:
        return candidate_id

    source_path = metadata.get("source_path", "")
    chunk_id = metadata.get("chunk_id", "")
    if source_path or chunk_id:
        return f"{source_path}::{chunk_id}"

    return ""


def _fallback_bm25_scores(tokenized_corpus: list[list[str]], query_tokens: list[str]) -> list[float]:
    query_terms = set(query_tokens)
    if not query_terms:
        return [0.0 for _ in tokenized_corpus]

    scores: list[float] = []
    for document_tokens in tokenized_corpus:
        if not document_tokens:
            scores.append(0.0)
            continue

        token_counts: dict[str, int] = {}
        for token in document_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        score = sum(token_counts.get(term, 0) for term in query_terms)
        scores.append(float(score) / max(len(document_tokens), 1))

    return scores


def load_all_indexed_documents() -> list[dict]:
    collection = get_collection()
    if collection is None:
        return []

    try:
        results = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"BM25 corpus load failed: {exc}")
        return []

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    indexed_documents: list[dict] = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        candidate_id = ids[index] if index < len(ids) else ""
        if not document:
            continue

        indexed_documents.append(
            {
                "id": candidate_id,
                "document": document,
                "metadata": metadata,
            }
        )

    return indexed_documents


def bm25_search(
    question: str,
    top_k: int = BM25_TOP_K,
    indexed_documents: list[dict] | None = None,
) -> list[dict]:
    documents = indexed_documents if indexed_documents is not None else load_all_indexed_documents()
    if not documents:
        return []

    query_tokens = tokenize_for_bm25(question)
    if not query_tokens:
        return []

    tokenized_corpus = [tokenize_for_bm25(item.get("document", "")) for item in documents]
    if BM25Okapi is not None:
        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = [float(score) for score in bm25.get_scores(query_tokens)]
    else:
        raw_scores = _fallback_bm25_scores(tokenized_corpus, query_tokens)

    normalized = normalize_scores(raw_scores)
    ranked: list[dict] = []
    for item, score in zip(documents, normalized):
        if score <= 0:
            continue

        ranked.append(
            {
                "id": item.get("id", ""),
                "document": item.get("document", ""),
                "metadata": item.get("metadata", {}),
                "bm25_score": score,
            }
        )

    ranked.sort(key=lambda item: item["bm25_score"], reverse=True)
    return ranked[:top_k]


def _merge_candidate(candidates: dict[str, dict], candidate: dict, source: str) -> None:
    metadata = candidate.get("metadata", {})
    candidate_id = candidate.get("id", "")
    key = _candidate_key(candidate_id, metadata) or candidate.get("document", "")
    if not key:
        return

    existing = candidates.get(key)
    if existing is None:
        sources = {source}
        candidates[key] = {
            "id": candidate_id,
            "document": candidate.get("document", ""),
            "metadata": metadata,
            "rerank_score": None,
            "keyword_score": candidate.get("keyword_score", 0.0),
            "vector_rank_score": candidate.get("vector_rank_score", 0.0),
            "bm25_score": candidate.get("bm25_score", 0.0),
            "final_score": 0.0,
            "_retrieval_sources": sources,
            "retrieval_source": ",".join(sorted(sources)),
        }
        return

    existing["keyword_score"] = max(existing.get("keyword_score", 0.0), candidate.get("keyword_score", 0.0))
    existing["vector_rank_score"] = max(
        existing.get("vector_rank_score", 0.0),
        candidate.get("vector_rank_score", 0.0),
    )
    existing["bm25_score"] = max(existing.get("bm25_score", 0.0), candidate.get("bm25_score", 0.0))
    existing_sources = existing.setdefault("_retrieval_sources", set())
    existing_sources.add(source)
    existing["retrieval_source"] = ",".join(sorted(existing_sources))


def retrieve_docs(question: str) -> list[dict]:
    collection = get_collection()
    if collection is None:
        return []

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
    ids = results.get("ids", [[]])[0]
    candidates_by_key: dict[str, dict] = {}

    for rank, (document, metadata) in enumerate(zip(documents, metadatas)):
        candidate_id = ids[rank] if rank < len(ids) else ""
        _merge_candidate(
            candidates_by_key,
            {
                "id": candidate_id,
                "document": document,
                "metadata": metadata,
                "keyword_score": keyword_score(question, document, metadata),
                "vector_rank_score": _vector_rank_score(rank, len(documents)),
            },
            "vector",
        )

    if BM25_ENABLED:
        for candidate in bm25_search(question, BM25_TOP_K):
            candidate["keyword_score"] = keyword_score(
                question,
                candidate.get("document", ""),
                candidate.get("metadata", {}),
            )
            _merge_candidate(candidates_by_key, candidate, "bm25")

    candidates = list(candidates_by_key.values())
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
                candidate["final_score"] = (
                    RERANK_WEIGHT * rerank_score
                    + BM25_WEIGHT * candidate["bm25_score"]
                    + KEYWORD_WEIGHT * candidate["keyword_score"]
                )
        except Exception as exc:
            print(f"Reranker failed; fallback scoring will be used: {exc}")
            for candidate in candidates:
                candidate["final_score"] = (
                    candidate["vector_rank_score"]
                    + BM25_WEIGHT * candidate["bm25_score"]
                    + KEYWORD_WEIGHT * candidate["keyword_score"]
                )
    else:
        for candidate in candidates:
            candidate["final_score"] = (
                candidate["vector_rank_score"]
                + BM25_WEIGHT * candidate["bm25_score"]
                + KEYWORD_WEIGHT * candidate["keyword_score"]
            )

    candidates.sort(key=lambda item: item["final_score"], reverse=True)
    for candidate in candidates:
        candidate.pop("_retrieval_sources", None)
    return candidates[:RERANK_TOP_K]


def search_docs(question: str) -> str:
    return format_context(retrieve_docs(question))
