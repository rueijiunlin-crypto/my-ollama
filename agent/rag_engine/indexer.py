import time
from datetime import datetime
from pathlib import Path

from config import COLLECTION_NAME, DATA_DIRS, EMBEDDING_MODEL, MANIFEST_PATH
from rag_engine.chunker import _chunk_metadata, _chunk_strategy, chunk_document
from rag_engine.file_reader import read_single_file
from rag_engine.manifest import (
    calculate_file_sha256,
    create_empty_manifest,
    delete_file_chunks,
    load_manifest,
    save_manifest,
    scan_source_files,
)
from rag_engine.models import encode_text, get_collection


def clear_collection() -> None:
    try:
        collection = get_collection()
        existing = collection.get()
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
    except Exception as exc:
        print(f"清除舊索引時發生警告：{exc}")


def index_single_file(path: Path, file_sha256: str) -> dict | None:
    try:
        docs = read_single_file(path)
    except Exception as exc:
        print(f"讀取失敗：{path}")
        print(f"原因：{exc}")
        return None

    stat = path.stat()
    chunk_ids: list[str] = []
    chunk_index = 0

    for doc in docs:
        text = doc.get("text", "")
        if not text.strip():
            continue

        metadata = doc.get("metadata", {})
        file_type = metadata.get("file_type", "").lower()
        chunk_strategy = _chunk_strategy(file_type, text)
        chunks = chunk_document(text, file_type)
        if not chunks:
            continue

        search_start = 0

        for chunk in chunks:
            chunk_id = f"{file_sha256[:16]}_{chunk_index}"

            print("正在建立向量：")
            print(path.name)
            print(f"strategy: {chunk_strategy}")

            start_index = text.find(chunk[:80], search_start)
            if start_index == -1:
                start_index = search_start
            search_start = start_index + max(len(chunk) - 150, 1)

            embedding = encode_text(chunk)
            chunk_metadata = _chunk_metadata(doc, chunk, chunk_id, start_index, chunk_strategy)

            get_collection().add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[chunk_metadata],
            )

            chunk_ids.append(chunk_id)
            chunk_index += 1

    if not chunk_ids:
        print(f"檔案沒有可索引內容，已略過：{path}")
        return None

    return {
        "source_path": str(path),
        "sha256": file_sha256,
        "mtime": stat.st_mtime,
        "file_size": stat.st_size,
        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_ids": chunk_ids,
    }


def build_index(full_rebuild: bool = False) -> None:
    start_time = time.perf_counter()

    if full_rebuild:
        clear_collection()
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
        elapsed = time.perf_counter() - start_time

        print("\n完整索引重建完成")
        print(f"索引檔案數：{indexed_files}")
        print(f"略過檔案數：{skipped_files}")
        print(f"Chunks：{chunk_count}")
        print(f"耗時：{elapsed:.1f} 秒")
        return

    manifest_exists = MANIFEST_PATH.exists()
    if not manifest_exists:
        try:
            existing_count = get_collection().count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            print("偵測到尚未建立 manifest，但 ChromaDB 可能已有舊索引。")
            print("建議先執行「完整重建索引」。")
            return

    manifest = load_manifest()

    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        print("目前 Embedding 模型與 manifest 不一致。")
        print("建議執行完整重建索引。")
        return

    if manifest.get("collection_name") != COLLECTION_NAME:
        print("目前 ChromaDB collection 與 manifest 不一致。")
        print("建議執行完整重建索引。")
        return

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
            delete_file_chunks(entry.get("chunk_ids", []))
            del manifest_files[source_path]
            deleted_file_count += 1

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

        is_update = old_entry is not None
        if old_entry:
            delete_file_chunks(old_entry.get("chunk_ids", []))

        try:
            entry = index_single_file(path, file_sha256)
        except Exception as exc:
            print(f"索引檔案失敗：{path}")
            print(f"原因：{exc}")
            entry = None

        if entry is None:
            if is_update:
                manifest_files.pop(path_key, None)
            continue

        manifest_files[path_key] = entry
        changed_chunk_count += len(entry.get("chunk_ids", []))
        if is_update:
            updated_file_count += 1
        else:
            new_file_count += 1

    manifest["embedding_model"] = EMBEDDING_MODEL
    manifest["collection_name"] = COLLECTION_NAME
    manifest["data_dirs"] = [str(path) for path in DATA_DIRS]
    save_manifest(manifest)

    elapsed = time.perf_counter() - start_time

    print("\n索引更新完成")
    print(f"新增檔案數：{new_file_count}")
    print(f"更新檔案數：{updated_file_count}")
    print(f"刪除檔案數：{deleted_file_count}")
    print(f"跳過未變更檔案數：{skipped_unchanged_count}")
    print(f"Chunks 新增/更新數：{changed_chunk_count}")
    print(f"耗時：{elapsed:.1f} 秒")
