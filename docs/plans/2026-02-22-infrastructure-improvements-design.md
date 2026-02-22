# gemini-mcp-server infrastructure improvements

## Context

gemini-mcp-server wraps the Gemini CLI as an MCP tool for Claude Code. Its core value: file contents are resolved server-side and piped to Gemini via stdin, so they never enter Claude's context window. Claude Code passes string references (paths, globs), the server reads files and handles the subprocess call.

The current architecture works well. These improvements make file context assembly easier and give Claude more information about what happened during a query.

## Changes

### 1. Directory parameter

Add `directories: list[str] | None` to `gemini_query`. The server recursively walks each directory and discovers files, which feed into the existing deduplication and byte-budget logic alongside `files` and `glob_patterns`.

No depth limit. The existing 500-file cap (`max_files`) and 10MB cap (`max_bytes`) are the safety net.

**files.py changes:**

Add `resolve_directories()` that walks directory trees and returns file paths. Integrate it into `resolve_files()` so directories, explicit files, and glob patterns all merge into the same deduplicated list.

```python
def resolve_files(
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    directories: list[str] | None = None,  # new
    max_files: int = 500,
) -> list[str]:
```

Directory walking uses `os.walk()`. All discovered paths go through the same `os.path.abspath()` deduplication and `os.path.isfile()` check.

**server.py changes:**

Add `directories` parameter to `gemini_query` tool and pass it through to `resolve_files()`.

```python
@mcp.tool
async def gemini_query(
    prompt: str,
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    directories: list[str] | None = None,  # new
    model: str | None = None,
    timeout: int = 120,
) -> str:
```

### 2. Response metadata

The Gemini CLI returns JSON with a `stats` object containing model info, token counts, and latency. Currently `_call_gemini` extracts only `data.get("response")` and discards everything else.

Surface this metadata so Claude knows which model answered and how many tokens were used.

**gemini.py changes:**

Change `_call_gemini` to return stats alongside the response text. Change `run_gemini` to format metadata into the response.

The stats object from the CLI looks like:

```json
{
  "stats": {
    "models": {
      "gemini-3-flash-preview": {
        "tokens": {
          "input": 8421,
          "candidates": 1333,
          "total": 10184
        }
      }
    }
  }
}
```

Append a metadata footer to the response:

```
[response text]

---
Model: gemini-3-flash-preview (fallback from gemini-3-pro-preview)
Tokens: 8421 input / 1333 output
```

When no fallback occurred, omit the parenthetical. When stats aren't available (e.g. non-JSON output), omit the footer entirely.

## Files changed

| File | Change |
|------|--------|
| `src/gemini_mcp/files.py` | Add `directories` param to `resolve_files()`, directory walking logic |
| `src/gemini_mcp/gemini.py` | Extract stats from CLI JSON, format metadata footer |
| `src/gemini_mcp/server.py` | Add `directories` param to `gemini_query` tool |
| `tests/test_files.py` | Tests for directory resolution, deduplication with mixed inputs |
| `tests/test_gemini.py` | Tests for stats extraction and metadata formatting |
| `tests/test_server.py` | Tests for directory parameter handling |

## What doesn't change

- The `gemini -p` subprocess call
- stdin piping of file context
- Model fallback chain and retry logic
- Error classification (fallback vs retriable patterns)
- XML-tagged context format
- Safety limits (500 files, 10MB)
