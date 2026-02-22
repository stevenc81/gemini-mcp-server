# CLAUDE.md

## Commands

```shell
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_files.py -v

# Run the MCP server locally
uv run gemini-mcp-server
```

## Architecture

This is a FastMCP server that wraps Gemini CLI as an MCP tool for Claude Code.

**Key modules in `src/gemini_mcp/`:**

- **server.py** — FastMCP app with one `@mcp.tool`: `gemini_query`. Accepts file paths, glob patterns, directories, model override, timeout, and `session_id` for multi-turn conversations. Resolves files, assembles context, delegates to `run_gemini`. Response includes metadata footer (model, tokens, session ID, skipped file count).
- **gemini.py** — Runs `gemini -p <prompt> -o json` as a subprocess. Pipes file context via stdin to leverage Gemini's large context window. Supports `--resume <session_id>` for conversation continuity (dropped on model fallback). Parses JSON response, extracts stats (model, tokens, session ID) into a metadata footer.
- **files.py** — `resolve_files()` takes explicit paths, glob patterns, and directories, returns `(file_list, skipped_count)`. Directories are walked recursively with junk directory pruning (`_SKIP_DIRS`). Binary/junk files are filtered via `_should_skip_file()` using `_SKIP_EXTENSIONS` and `_SKIP_FILES` (applied to globs and directories, not explicit files). `read_files_as_context()` reads them into XML-tagged blocks.

**Data flow:**
`Claude Code` → `server.py (gemini_query tool)` → `files.py (resolve + read)` → `gemini.py (subprocess)` → `Gemini CLI` → response text back to Claude Code
