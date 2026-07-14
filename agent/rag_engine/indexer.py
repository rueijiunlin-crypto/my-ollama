import time
from datetime import datetime
from pathlib import Path

from config import (
    CHROMA_BATCH_SIZE,
    COLLECTION_NAME,
    DATA_DIRS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    INDEX_SCHEMA_VERSION,
    MANIFEST_PATH,
)
from rag_engine.chunker import (
    _chunk_metadata,
    _chunk_strategy,
    create_parent_child_chunks,
)
from rag_engine.file_reader import read_single_file
from rag_engine.manifest import (
    calculate_file_sha256,
    create_empty_manifest,
    delete_file_chunks,
    load_manifest,
    save_manifest,
    scan_source_files,
)
from rag_engine.models import encode_texts, get_collection


def _invalidate_retrieval_cache() -> None:
    # 延遲匯入可避免 indexer 與 retriever 形成循環 import。
    try:
        from rag_engine.retriever import invalidate_bm25_cache

        invalidate_bm25_cache()
    except (ImportError, AttributeError):
        pass


def clear_collection() -> bool:
    """完整重建前清除舊索引；失敗時必須阻止後續重建。"""
    try:
        collection = get_collection()
        if collection is None:
            raise RuntimeError("ChromaDB collection 尚未載入。")
        existing = collection.get()
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return True
    except Exception as exc:
        print(f"清除舊索引失敗，已中止完整重建：{exc}")
        return False


def _prepare_single_file(path: Path, file_sha256: str) -> tuple[dict, list[dict]] | None:
    """先在記憶體完成 Chunk 與 Metadata，尚不修改 ChromaDB。"""
    try:
        docs = read_single_file(path)
    except Exception as exc:
        print(f"讀取失敗：{path}")
        print(f"原因：{exc}")
        return None

    stat = path.stat()
    records: list[dict] = []
    for doc_index, doc in enumerate(docs):
        text = doc.get("text", "")
        if not text.strip():
            continue

        metadata = doc.get("metadata", {})
        file_type = metadata.get("file_type", "").lower()
        chunk_strategy = _chunk_strategy(file_type, text)
        parent_children = create_parent_child_chunks(
            text,
            file_type,
            f"{file_sha256[:16]}_d{doc_index}",
        )
        search_start = 0

        for item in parent_children:
            child = item["child_text"]
            child_id = f"{item['parent_id']}_c{item['child_index']}"
            start_index = text.find(child[:80], search_start)
            if start_index == -1:
                start_index = search_start
            search_start = start_index + max(len(child) - 100, 1)

            chunk_metadata = _chunk_metadata(
                doc,
                child,
                child_id,
                start_index,
                chunk_strategy,
            )
            chunk_metadata.update(
                {
                    "parent_id": item["parent_id"],
                    "parent_index": item["parent_index"],
                    "child_index": item["child_index"],
                    "parent_text": item["parent_text"],
                }
            )
            records.append(
                {
                    "id": child_id,
                    "document": child,
                    "metadata": chunk_metadata,
                }
            )

    if not records:
        print(f"檔案沒有可索引內容，已略過：{path}")
        return None

    entry = {
        "source_path": str(path),
        "sha256": file_sha256,
        "mtime": stat.st_mtime,
        "file_size": stat.st_size,
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_ids": [record["id"] for record in records],
        "parent_count": len({record["metadata"]["parent_id"] for record in records}),
    }
    return entry, records


def _add_records(records: list[dict]) -> bool:
    """批次向量化並寫入；失敗時清掉本次已加入的資料。"""
    if not records:
        return False

    collection = get_collection()
    if collection is None:
        raise RuntimeError("ChromaDB collection 尚未載入。")

    added_ids: list[str] = []
    try:
        for start in range(0, len(records), CHROMA_BATCH_SIZE):
            batch = records[start:start + CHROMA_BATCH_SIZE]
            documents = [item["document"] for item in batch]
            embeddings = encode_texts(documents, batch_size=EMBEDDING_BATCH_SIZE)
            ids = [item["id"] for item in batch]
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=[item["metadata"] for item in batch],
            )
            added_ids.extend(ids)
        return True
    except Exception as exc:
        print(f"批次寫入索引失敗：{exc}")
        if added_ids:
            delete_file_chunks(added_ids)
        return False


def index_single_file(path: Path, file_sha256: str) -> dict | None:
    prepared = _prepare_single_file(path, file_sha256)
    if prepared is None:
        return None

    entry, records = prepared
    print(f"正在建立向量：{path}")
    print(f"Child chunks：{len(records)}；Parents：{entry['parent_count']}")
    if not _add_records(records):
        return None
    return entry


def build_index(full_rebuild: bool = False) -> bool:
    start_time = time.perf_counter()

    if full_rebuild:
        if not clear_collection():
            return False

        manifest = create_empty_manifest()
        source_files = scan_source_files(DATA_DIRS)
        indexed_files = 0
        skipped_files = 0
        chunk_count = 0

        for path in source_files:
            try:
                file_sha256 = calculate_file_sha256(path)
                entry = index_single_file(path, file_sha256)
            except Exception as exc:
                print(f"索引檔案失敗：{path}")
                print(f"原因：{exc}")
                entry = None

            if entry is None:
                skipped_files += 1
                continue

            manifest["files"][str(path)] = entry
            indexed_files += 1
            chunk_count += len(entry.get("chunk_ids", []))

        save_manifest(manifest)
        _invalidate_retrieval_cache()
        elapsed = time.perf_counter() - start_time

        print("\n完整索引重建完成")
        print(f"索引檔案數：{indexed_files}")
        print(f"略過檔案數：{skipped_files}")
        print(f"Child chunks：{chunk_count}")
        print(f"耗時：{elapsed:.1f} 秒")
        return True

    manifest_exists = MANIFEST_PATH.exists()
    if not manifest_exists:
        try:
            existing_count = get_collection().count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            print("偵測到尚未建立 manifest，但 ChromaDB 可能已有舊索引。")
            print("建議先執行「完整重建索引」。")
            return False

    manifest = load_manifest()
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        print("目前 Embedding 模型與 manifest 不一致。")
        print("建議執行完整重建索引。")
        return False
    if manifest.get("collection_name") != COLLECTION_NAME:
        print("目前 ChromaDB collection 與 manifest 不一致。")
        print("建議執行完整重建索引。")
        return False
    if manifest.get("index_schema_version") != INDEX_SCHEMA_VERSION:
        print("索引資料格式已更新，請先執行完整重建索引。")
        return False

    source_files = scan_source_files(DATA_DIRS)
    current_paths = {str(path) for path in source_files}
    manifest_files = manifest.setdefault("files", {})
    new_file_count = 0
    updated_file_count = 0
    deleted_file_count = 0
    skipped_unchanged_count = 0
    changed_chunk_count = 0

    for source_path in list(manifest_files.keys()):
        if source_path not in current_paths:
            entry = manifest_files[source_path]
            if delete_file_chunks(entry.get("chunk_ids", [])):
                del manifest_files[source_path]
                deleted_file_count += 1
            else:
                print(f"保留 manifest 紀錄，因舊索引刪除失敗：{source_path}")

    for path in source_files:
        path_key = str(path)
        try:
            file_sha256 = calculate_file_sha256(path)
        except Exception as exc:
            print(f"計算檔案 sha256 失敗：{path}")
            print(f"原因：{exc}")
            continue

        old_entry = manifest_files.get(path_key)
        if old_entry and old_entry.get("sha256") == file_sha256:
            skipped_unchanged_count += 1
            continue

        # 原子式更新：先建立新索引，成功後才刪除舊索引。
        try:
            new_entry = index_single_file(path, file_sha256)
        except Exception as exc:
            print(f"索引檔案失敗：{path}")
            print(f"原因：{exc}")
            new_entry = None

        if new_entry is None:
            print(f"保留舊索引：{path}") if old_entry else None
            continue

        if old_entry and not delete_file_chunks(old_entry.get("chunk_ids", [])):
            print(f"舊索引刪除失敗，正在回復本次新索引：{path}")
            delete_file_chunks(new_entry.get("chunk_ids", []))
            continue

        manifest_files[path_key] = new_entry
        changed_chunk_count += len(new_entry.get("chunk_ids", []))
        if old_entry:
            updated_file_count += 1
        else:
            new_file_count += 1

    manifest["embedding_model"] = EMBEDDING_MODEL
    manifest["collection_name"] = COLLECTION_NAME
    manifest["index_schema_version"] = INDEX_SCHEMA_VERSION
    manifest["data_dirs"] = [str(path) for path in DATA_DIRS]
    save_manifest(manifest)
    if new_file_count or updated_file_count or deleted_file_count:
        _invalidate_retrieval_cache()

    elapsed = time.perf_counter() - start_time
    print("\n索引更新完成")
    print(f"新增檔案數：{new_file_count}")
    print(f"更新檔案數：{updated_file_count}")
    print(f"刪除檔案數：{deleted_file_count}")
    print(f"跳過未變更檔案數：{skipped_unchanged_count}")
    print(f"Child chunks 新增/更新數：{changed_chunk_count}")
    print(f"耗時：{elapsed:.1f} 秒")
    return True
