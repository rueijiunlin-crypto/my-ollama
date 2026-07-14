import hashlib
import json
from pathlib import Path

from config import (
    COLLECTION_NAME,
    DATA_DIRS,
    EMBEDDING_MODEL,
    INDEX_SCHEMA_VERSION,
    MANIFEST_PATH,
    SUPPORTED_EXTENSIONS,
)
from rag_engine.models import get_collection
from rag_engine.path_filter import iter_source_files


def calculate_file_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def create_empty_manifest() -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "collection_name": COLLECTION_NAME,
        "data_dirs": [str(path) for path in DATA_DIRS],
        "files": {},
    }


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return create_empty_manifest()

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception as exc:
        backup_path = MANIFEST_PATH.with_name("index_manifest.bak.json")
        try:
            backup_path.write_text(MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"manifest 損壞，已備份至：{backup_path}")
        except Exception as backup_exc:
            print(f"manifest 損壞，且備份失敗：{backup_exc}")
        print(f"將建立新的 manifest。原因：{exc}")
        return create_empty_manifest()

    if not isinstance(manifest, dict):
        print("manifest 格式不正確，將建立新的 manifest。")
        return create_empty_manifest()

    manifest.setdefault("embedding_model", EMBEDDING_MODEL)
    manifest.setdefault("index_schema_version", 1)
    manifest.setdefault("collection_name", COLLECTION_NAME)
    manifest.setdefault("data_dirs", [str(path) for path in DATA_DIRS])
    manifest.setdefault("files", {})
    return manifest


def save_manifest(manifest: dict) -> None:
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(MANIFEST_PATH)


def scan_source_files(folders: list[Path]) -> list[Path]:
    files: list[Path] = []

    for folder in folders:
        if not folder.exists():
            print(f"找不到資料夾：{folder}")
            continue

        for path in iter_source_files(folder):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)

    return sorted(files, key=lambda item: str(item).lower())


def delete_file_chunks(chunk_ids: list[str]) -> bool:
    if not chunk_ids:
        return True

    try:
        get_collection().delete(ids=chunk_ids)
        return True
    except Exception as exc:
        print(f"刪除舊 chunks 時發生警告：{exc}")
        return False
