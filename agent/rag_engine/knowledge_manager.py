from collections import Counter
from pathlib import Path

from config import COLLECTION_NAME, DATA_DIRS, EMBEDDING_MODEL
from rag_engine.manifest import load_manifest
from rag_engine.models import get_collection


def get_knowledge_status() -> dict:
    manifest = load_manifest()
    files = manifest.get("files", {})
    file_types: Counter[str] = Counter()
    root_sources: Counter[str] = Counter()
    indexed_times: list[str] = []
    parent_count = 0

    for source_path, entry in files.items():
        path = Path(source_path)
        file_types[path.suffix.lower() or "無副檔名"] += 1
        indexed_times.append(str(entry.get("indexed_at", "")))
        parent_count += int(entry.get("parent_count", 0))

        matched_source = "其他"
        for root in DATA_DIRS:
            try:
                path.resolve().relative_to(root.resolve())
                matched_source = root.name
                break
            except (OSError, ValueError):
                continue
        root_sources[matched_source] += 1

    collection = get_collection()
    chunk_count = collection.count() if collection is not None else 0
    return {
        "file_count": len(files),
        "chunk_count": chunk_count,
        "parent_count": parent_count,
        "file_types": dict(sorted(file_types.items())),
        "root_sources": dict(sorted(root_sources.items())),
        "last_indexed_at": max(indexed_times, default="無"),
        "embedding_model": manifest.get("embedding_model", EMBEDDING_MODEL),
        "collection_name": manifest.get("collection_name", COLLECTION_NAME),
        "data_dirs": manifest.get("data_dirs", [str(path) for path in DATA_DIRS]),
    }


def format_knowledge_status(status: dict) -> str:
    lines = [
        "\n=== 知識庫狀態 ===",
        f"索引文件數：{status['file_count']}",
        f"Parent 數量：{status['parent_count']}",
        f"Child Chunk 數量：{status['chunk_count']}",
        f"最近索引時間：{status['last_indexed_at']}",
        f"Embedding 模型：{status['embedding_model']}",
        f"Collection：{status['collection_name']}",
        "資料來源：",
    ]
    lines.extend(
        f"- {source}: {count} 個文件"
        for source, count in status["root_sources"].items()
    )
    lines.append("文件類型：")
    lines.extend(
        f"- {file_type}: {count}"
        for file_type, count in status["file_types"].items()
    )
    return "\n".join(lines)


def list_indexed_files(search_text: str = "") -> list[str]:
    files = sorted(load_manifest().get("files", {}).keys(), key=str.casefold)
    if search_text:
        needle = search_text.casefold()
        files = [path for path in files if needle in path.casefold()]
    return files


def format_knowledge_manager_help() -> str:
    return """\
\n=== Knowledge Manager 使用指南 ===
1：查看文件數、Parent／Child Chunk 數、來源、格式與最近索引時間。
2：列出 Manifest 中全部已索引文件的完整路徑。
3：依「檔名或路徑」搜尋已索引文件。
   選擇 3 後輸入部分關鍵字即可，不分英文大小寫。
   例如輸入 clip，可找出檔名或路徑中包含 CLIP／clip 的文件。
   例如輸入 Week03，可找出位於 Week03 路徑下的文件。
   此功能只搜尋檔名與路徑，不會搜尋文件內文。
help 或 ?：再次顯示本指南。
back：回到主選單。"""


def run_knowledge_manager() -> None:
    while True:
        print("\n=== Knowledge Manager ===")
        print("1. 查看知識庫狀態")
        print("2. 列出已索引文件")
        print("3. 搜尋已索引檔名或路徑（部分關鍵字）")
        print("help. 使用指南")
        print("back. 回主選單")
        command = input("請選擇功能：").strip()

        if command == "1":
            print(format_knowledge_status(get_knowledge_status()))
        elif command == "2":
            files = list_indexed_files()
            print("\n".join(files) if files else "目前沒有已索引文件。")
        elif command == "3":
            print("輸入部分檔名或路徑即可；此功能不搜尋文件內文。")
            keyword = input("請輸入檔名或路徑關鍵字：").strip()
            if not keyword:
                print("關鍵字不可為空白。")
                continue
            files = list_indexed_files(keyword)
            print("\n".join(files) if files else "找不到符合的已索引文件。")
        elif command.lower() in {"help", "?"}:
            print(format_knowledge_manager_help())
        elif command.lower() == "back":
            return
        else:
            print("無效選項，請重新輸入。")
