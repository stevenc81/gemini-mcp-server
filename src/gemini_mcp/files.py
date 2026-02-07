import glob as globmod
import os


def resolve_files(
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    max_files: int = 500,
) -> list[str]:
    """Resolve explicit file paths and glob patterns into a deduplicated list of existing file paths."""
    resolved: list[str] = []
    seen: set[str] = set()

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
                resolved.append(abs_path)
                seen.add(abs_path)

    return resolved


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
        try:
            with open(path, "r") as f:
                content = f.read()
            file_bytes = len(content.encode("utf-8"))
            if total_bytes + file_bytes > max_bytes and included > 0:
                break
            blocks.append(f'<file path="{path}">\n{content}\n</file>')
            total_bytes += file_bytes
            included += 1
        except (OSError, UnicodeDecodeError):
            blocks.append(f'<file path="{path}" error="could not read file" />')
            included += 1

    if included < len(file_paths):
        blocks.append(
            f"<note>Context truncated: reached {max_bytes} byte limit. "
            f"{included} of {len(file_paths)} files included.</note>"
        )

    return "\n\n".join(blocks)
