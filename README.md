# gemini-mcp-server

An MCP server that lets Claude Code call Gemini CLI as a tool. Leverages Gemini's large context window for research, code review, and analysis tasks.

## How it works

Claude Code sends lightweight instructions (a prompt, file paths, glob patterns). The server resolves files and reads them server-side, then pipes everything to `gemini -p` via stdin. Gemini's response comes back as text to Claude Code.

This means file contents never consume Claude's context window — only Gemini's (up to 1M+ tokens).

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated

## Setup

1. Install and authenticate the [Gemini CLI](https://github.com/google-gemini/gemini-cli):

```bash
npm install -g @google/gemini-cli
gemini  # follow the auth prompts
```

2. Clone this repo:

```bash
git clone https://github.com/stevenc81/gemini-mcp-server.git
```

3. Verify Gemini CLI works:

```bash
gemini -p "hello" -o json
```

4. Register the server with Claude Code:

```bash
claude mcp add gemini --transport stdio --scope user -- uv run --project /path/to/gemini-mcp-server gemini-mcp-server
```

Replace `/path/to/gemini-mcp-server` with the absolute path to where you cloned the repo.

You should now be able to ask Claude to use the `gemini_query` tool.

## Usage

Once registered, Claude Code can call the `gemini_query` tool:

```
# Simple question
"Ask Gemini what the best approach for rate limiting in Go is"

# Code review with file context
"Have Gemini review src/auth.ts and src/auth.test.ts for security issues"

# Large codebase analysis
"Ask Gemini to find inconsistent error handling across src/**/*.ts"

# Use a specific model
"Ask Gemini using gemini-2.5-flash to summarize this file"
```

### Parameters

- `prompt` (string, required) — The instruction or question for Gemini
- `files` (string[], optional) — Absolute file paths to include as context
- `glob_patterns` (string[], optional) — Glob patterns resolved server-side
- `model` (string, optional) — Override the model. Skips the fallback chain and uses this model only
- `timeout` (int, optional) — Max seconds to wait. Default 120

### Model fallback

When no model is specified, the server tries these in order until one succeeds:

1. `gemini-3-pro-preview`
2. `gemini-3-flash-preview`
3. `gemini-2.5-pro`
4. `gemini-2.5-flash`

If you specify a model explicitly (e.g., `model="gemini-2.5-flash"`), it uses that model directly with no fallback.

### Safety limits

- Max 500 files per query (configurable via `max_files` in `resolve_files`)
- Max 10MB total context (configurable via `max_bytes` in `read_files_as_context`)
- File sizes are checked with `os.path.getsize()` before reading to prevent OOM
- File paths are displayed as relative paths in context sent to Gemini

### Robustness

- JSON parsing handles CLI warnings and non-JSON output mixed with the response
- Blocking file I/O is offloaded to threads via `asyncio.to_thread`
- Subprocess uses `create_subprocess_exec` (no shell injection risk)
- Timeouts prevent hung Gemini CLI processes

## Development

```bash
# Install dependencies
uv sync

# Run tests (24 tests)
uv run pytest -v

# Run the server locally
uv run gemini-mcp-server
```
