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

- **server.py** — FastMCP app with one `@mcp.tool`: `gemini_query`. Resolves files, assembles context, delegates to `run_gemini`.
- **gemini.py** — Runs `gemini -p <prompt> -o json` as a subprocess. Pipes file context via stdin to leverage Gemini's large context window. Parses JSON response.
- **files.py** — `resolve_files()` takes explicit paths and glob patterns, returns deduplicated list of existing file paths. `read_files_as_context()` reads them into XML-tagged blocks.

**Data flow:**
`Claude Code` → `server.py (gemini_query tool)` → `files.py (resolve + read)` → `gemini.py (subprocess)` → `Gemini CLI` → response text back to Claude Code
