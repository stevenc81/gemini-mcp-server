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

- **server.py** — FastMCP app with one `@mcp.tool`: `gemini_query`. Accepts file paths, glob patterns, and directories. Resolves files, assembles context, delegates to `run_gemini`. Response includes metadata footer (model used, token counts).
- **gemini.py** — Runs `gemini -p <prompt> -o json` as a subprocess. Pipes file context via stdin to leverage Gemini's large context window. Parses JSON response and extracts stats (model, tokens) into a metadata footer.
- **files.py** — `resolve_files()` takes explicit paths, glob patterns, and directories, returns deduplicated list of existing file paths. Directories are walked recursively with junk directory pruning (`_SKIP_DIRS`). `read_files_as_context()` reads them into XML-tagged blocks.

**Data flow:**
`Claude Code` → `server.py (gemini_query tool)` → `files.py (resolve + read)` → `gemini.py (subprocess)` → `Gemini CLI` → response text back to Claude Code
