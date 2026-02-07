from fastmcp import FastMCP

from gemini_mcp.files import resolve_files, read_files_as_context
from gemini_mcp.gemini import run_gemini

mcp = FastMCP("Gemini")


@mcp.tool
async def gemini_query(
    prompt: str,
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """Send a query to Gemini CLI and return its response.

    Use this to leverage Gemini's large context window for tasks like:
    - Analyzing large codebases (pass glob patterns to include many files)
    - Getting a second opinion on code, architecture, or bugs
    - Research tasks that benefit from Gemini's training data
    - Reviewing code across many files at once

    File contents are loaded server-side and piped to Gemini via stdin,
    so they don't consume Claude's context window.

    Args:
        prompt: The instruction or question for Gemini.
        files: Optional list of absolute file paths to include as context.
        glob_patterns: Optional glob patterns (e.g., "src/**/*.py") resolved server-side.
        model: Optional Gemini model (e.g., "gemini-2.5-pro"). Uses Gemini CLI default if not set.
        timeout: Max seconds to wait for Gemini response. Default 120.
    """
    file_paths = resolve_files(files=files, glob_patterns=glob_patterns)
    context = read_files_as_context(file_paths)

    return await run_gemini(
        prompt=prompt,
        context=context,
        model=model,
        timeout=timeout,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
