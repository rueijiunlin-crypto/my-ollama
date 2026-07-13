import os
from pathlib import Path
from typing import Iterator

from config import EXCLUDED_DIR_NAMES


_EXCLUDED_DIR_NAMES_LOWER = {name.casefold() for name in EXCLUDED_DIR_NAMES}


def is_excluded_path(path: Path) -> bool:
    """判斷路徑是否位於排除目錄內（不分大小寫）。"""
    return any(part.casefold() in _EXCLUDED_DIR_NAMES_LOWER for part in path.parts)


def is_link_or_junction(path: Path) -> bool:
    """判斷路徑是否為符號連結或 Windows junction/reparse point。"""
    try:
        if path.is_symlink():
            return True

        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and is_junction():
            return True

        # Python 3.11 沒有 Path.is_junction()；Windows junction 具有
        # FILE_ATTRIBUTE_REPARSE_POINT (0x400)。
        if os.name == "nt":
            file_attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
            if file_attributes & 0x400:
                return True
    except OSError:
        # 無法安全檢查的路徑不要進入。
        return True

    return False


def iter_source_files(folder: Path) -> Iterator[Path]:
    """遞迴列出來源檔案，並在深入前排除目錄、連結及 junction。"""
    if is_excluded_path(folder) or is_link_or_junction(folder):
        return

    for root, dir_names, file_names in os.walk(folder, topdown=True, followlinks=False):
        root_path = Path(root)

        # 原地修改 dir_names，讓 os.walk 不會進入被排除的子樹。
        dir_names[:] = [
            name
            for name in dir_names
            if not is_excluded_path(root_path / name)
            and not is_link_or_junction(root_path / name)
        ]

        for file_name in file_names:
            path = root_path / file_name
            if not is_excluded_path(path) and not is_link_or_junction(path):
                yield path
