import glob as globmod
import os

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", ".env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", ".nuxt", "dist", "build", ".eggs",
}

_SKIP_EXTENSIONS = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp", ".webp",
    # Audio/video
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Compiled/binary
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".o", ".a", ".class",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Source maps
    ".map",
}

_SKIP_FILES = {".DS_Store"}


def _should_skip_file(filename: str) -> bool:
    """Return True if this file should be excluded from context."""
    if filename in _SKIP_FILES:
        return True
    _, ext = os.path.splitext(filename)
    return ext.lower() in _SKIP_EXTENSIONS


def resolve_files(
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    directories: list[str] | None = None,
    max_files: int = 500,
) -> tuple[list[str], int]:
    """Resolve explicit file paths, glob patterns, and directories into a deduplicated list of existing file paths.

    Returns a tuple of (resolved_paths, skipped_count). Files discovered via
    glob patterns or directory walks are filtered against ``_should_skip_file``;
    explicitly listed files are never filtered.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    skipped = 0

    for path in files or []:
        if len(resolved) >= max_files:
            break
        abs_path = os.path.abspath(path)
        if abs_path not in seen and os.path.isfile(abs_path):
            resolved.append(abs_path)
            seen.add(abs_path)

    for pattern in glob_patterns or []:
        if len(resolved) >= max_files:
            break
        for path in sorted(globmod.glob(pattern, recursive=True)):
            if len(resolved) >= max_files:
                break
            abs_path = os.path.abspath(path)
            if abs_path not in seen and os.path.isfile(abs_path):
                if _should_skip_file(os.path.basename(abs_path)):
                    skipped += 1
                    continue
                resolved.append(abs_path)
                seen.add(abs_path)

    for directory in directories or []:
        if len(resolved) >= max_files:
            break
        abs_dir = os.path.abspath(directory)
        if not os.path.isdir(abs_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(abs_dir):
            # Prune junk directories in-place so os.walk skips them.
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            if len(resolved) >= max_files:
                break
            for filename in sorted(filenames):
                if len(resolved) >= max_files:
                    break
                abs_path = os.path.join(dirpath, filename)
                if abs_path not in seen and os.path.isfile(abs_path):
                    if _should_skip_file(filename):
                        skipped += 1
                        continue
                    resolved.append(abs_path)
                    seen.add(abs_path)

    return resolved, skipped


def _display_path(path: str) -> str:
    """Convert absolute path to a shorter relative path when possible."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def read_files_as_context(
    file_paths: list[str],
    max_bytes: int = 10_000_000,
) -> str:
    """Read files and format them as XML-tagged context blocks."""
    if not file_paths:
        return ""

    blocks: list[str] = []
    total_bytes = 0
    included = 0
    for path in file_paths:
        display = _display_path(path)
        try:
            file_size = os.path.getsize(path)
            if total_bytes + file_size > max_bytes:
                break
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            blocks.append(f'<file path="{display}">\n{content}\n</file>')
            total_bytes += file_size
            included += 1
        except (OSError, UnicodeDecodeError):
            blocks.append(f'<file path="{display}" error="could not read file" />')
            included += 1

    if included < len(file_paths):
        blocks.append(
            f"<note>Context truncated: reached {max_bytes} byte limit. "
            f"{included} of {len(file_paths)} files included.</note>"
        )

    return "\n\n".join(blocks)
