# gemini-mcp-server

An MCP server that lets Claude Code call Gemini CLI as a tool. Leverages Gemini's large context window for research, code review, and analysis tasks.

## How it works

Claude Code sends lightweight instructions (a prompt, file paths, glob patterns). The server resolves files and reads them server-side, then pipes everything to `gemini -p` via stdin. Gemini's response comes back as text to Claude Code.

This means file contents never consume Claude's context window — only Gemini's.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated

## Setup

Register with Claude Code:

```bash
claude mcp add --transport stdio --scope user gemini-mcp uv run --project /path/to/gemini-mcp-server gemini-mcp-server
```

## Usage

Once registered, Claude Code can call the `gemini_query` tool:

```
# Simple question
"Ask Gemini what the best approach for rate limiting in Go is"

# Code review with file context
"Have Gemini review src/auth.ts and src/auth.test.ts for security issues"

# Large codebase analysis
"Ask Gemini to find inconsistent error handling across src/**/*.ts"
```

### Parameters

- `prompt` (string, required) — The instruction or question for Gemini
- `files` (string[], optional) — Absolute file paths to include as context
- `glob_patterns` (string[], optional) — Glob patterns resolved server-side
- `model` (string, optional) — Gemini model to use (e.g., `gemini-2.5-pro`)
- `timeout` (int, optional) — Max seconds to wait. Default 120

### Safety limits

- Max 500 files per query (configurable)
- Max 10MB total context (configurable)
- Files are size-checked before reading to prevent OOM

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -v

# Run the server locally
uv run gemini-mcp-server
```
