import glob as globmod
import os


def resolve_files(
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
) -> list[str]:
    """Resolve explicit file paths and glob patterns into a deduplicated list of existing file paths."""
    resolved: list[str] = []
    seen: set[str] = set()

    for path in files or []:
        abs_path = os.path.abspath(path)
        if abs_path not in seen and os.path.isfile(abs_path):
            resolved.append(abs_path)
            seen.add(abs_path)

    for pattern in glob_patterns or []:
        for path in sorted(globmod.glob(pattern, recursive=True)):
            abs_path = os.path.abspath(path)
            if abs_path not in seen and os.path.isfile(abs_path):
                resolved.append(abs_path)
                seen.add(abs_path)

    return resolved


def read_files_as_context(file_paths: list[str]) -> str:
    """Read files and format them as XML-tagged context blocks."""
    if not file_paths:
        return ""

    blocks: list[str] = []
    for path in file_paths:
        try:
            with open(path, "r") as f:
                content = f.read()
            blocks.append(f'<file path="{path}">\n{content}\n</file>')
        except (OSError, UnicodeDecodeError):
            blocks.append(f'<file path="{path}" error="could not read file" />')

    return "\n\n".join(blocks)
