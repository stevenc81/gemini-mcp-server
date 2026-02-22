import asyncio

from fastmcp import FastMCP

from gemini_mcp.files import resolve_files, read_files_as_context
from gemini_mcp.gemini import run_gemini

MODEL_FALLBACK_CHAIN = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

mcp = FastMCP("Gemini")


@mcp.tool
async def gemini_query(
    prompt: str,
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    directories: list[str] | None = None,
    model: str | None = None,
    timeout: int = 120,
    session_id: str | None = None,
) -> str:
    """Send a query to Gemini CLI and return its response.

    Use this to leverage Gemini's large context window for tasks like:
    - Analyzing large codebases (pass glob patterns to include many files)
    - Getting a second opinion on code, architecture, or bugs
    - Research tasks that benefit from Gemini's training data
    - Reviewing code across many files at once

    File contents are loaded server-side and piped to Gemini via stdin,
    so they don't consume Claude's context window.

    Defaults to gemini-3-pro-preview, falling back to gemini-3-flash-preview,
    gemini-2.5-pro, then gemini-2.5-flash if a model is unavailable.

    Args:
        prompt: The instruction or question for Gemini.
        files: Optional list of absolute file paths to include as context.
        glob_patterns: Optional glob patterns (e.g., "src/**/*.py") resolved server-side.
        directories: Optional list of directory paths to recursively include.
        model: Specific Gemini model to use (no fallback). If not set, tries the fallback chain.
        timeout: Max seconds to wait for Gemini response. Default 120.
        session_id: Optional session ID to resume a previous conversation.
    """
    file_paths, skipped_files = await asyncio.to_thread(resolve_files, files, glob_patterns, directories)
    context = await asyncio.to_thread(read_files_as_context, file_paths)

    return await run_gemini(
        prompt=prompt,
        context=context,
        model=model,
        models=None if model else MODEL_FALLBACK_CHAIN,
        timeout=timeout,
        skipped_files=skipped_files,
        session_id=session_id,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
