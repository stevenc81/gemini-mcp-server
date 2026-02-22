# Infrastructure improvements implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add directory parameter to `gemini_query` and surface CLI response metadata (model, tokens).

**Architecture:** Two independent changes to the existing pipeline. Directory support adds a new input path in `files.py` that merges into the existing dedup/budget logic. Response metadata extracts stats from the CLI's JSON output in `gemini.py` and appends a footer.

**Tech Stack:** Python 3.12+, FastMCP, pytest, pytest-asyncio

---

### Task 1: Directory resolution in files.py

**Files:**
- Modify: `src/gemini_mcp/files.py`
- Test: `tests/test_files.py`

**Step 1: Write the failing tests**

Add these tests to `tests/test_files.py`:

```python
def test_resolve_files_with_directories(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.py").write_text("root")
    (sub / "nested.py").write_text("nested")
    result = resolve_files(directories=[str(tmp_path)])
    abs_paths = [os.path.basename(p) for p in result]
    assert "root.py" in abs_paths
    assert "nested.py" in abs_paths


def test_resolve_files_directories_dedup_with_files(tmp_path):
    f = tmp_path / "dup.py"
    f.write_text("dup")
    result = resolve_files(
        files=[str(f)],
        directories=[str(tmp_path)],
    )
    assert result.count(str(f)) == 1


def test_resolve_files_directories_respects_max_files(tmp_path):
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text(f"content{i}")
    result = resolve_files(directories=[str(tmp_path)], max_files=3)
    assert len(result) == 3


def test_resolve_files_directories_skips_nonexistent():
    result = resolve_files(directories=["/nonexistent/directory"])
    assert result == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_files.py -v -k "directories"`
Expected: FAIL — `resolve_files()` doesn't accept `directories` parameter

**Step 3: Implement directory support in resolve_files**

In `src/gemini_mcp/files.py`, add `directories` parameter to `resolve_files()` and walk each directory with `os.walk()`:

```python
def resolve_files(
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    directories: list[str] | None = None,
    max_files: int = 500,
) -> list[str]:
    """Resolve explicit file paths, glob patterns, and directories into a deduplicated list of existing file paths."""
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

    for directory in directories or []:
        if len(resolved) >= max_files:
            break
        abs_dir = os.path.abspath(directory)
        if not os.path.isdir(abs_dir):
            continue
        for dirpath, _, filenames in os.walk(abs_dir):
            if len(resolved) >= max_files:
                break
            for filename in sorted(filenames):
                if len(resolved) >= max_files:
                    break
                abs_path = os.path.join(dirpath, filename)
                if abs_path not in seen:
                    resolved.append(abs_path)
                    seen.add(abs_path)

    return resolved
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_files.py -v`
Expected: All tests PASS (old and new)

**Step 5: Commit**

```bash
git add src/gemini_mcp/files.py tests/test_files.py
git commit -m "feat: add directory parameter to resolve_files"
```

---

### Task 2: Wire directories into gemini_query

**Files:**
- Modify: `src/gemini_mcp/server.py`
- Test: `tests/test_server.py`

**Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
@pytest.mark.asyncio
async def test_gemini_query_with_directories(tmp_path):
    (tmp_path / "app.py").write_text("print('app')")
    sub = tmp_path / "lib"
    sub.mkdir()
    (sub / "util.py").write_text("print('util')")
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Reviewed"
        result = await gemini_query.fn(
            prompt="Review",
            directories=[str(tmp_path)],
        )
        assert result == "Reviewed"
        call_kwargs = mock_run.call_args[1]
        assert "print('app')" in call_kwargs["context"]
        assert "print('util')" in call_kwargs["context"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py::test_gemini_query_with_directories -v`
Expected: FAIL — `gemini_query()` doesn't accept `directories`

**Step 3: Add directories parameter to gemini_query**

In `src/gemini_mcp/server.py`:

```python
@mcp.tool
async def gemini_query(
    prompt: str,
    files: list[str] | None = None,
    glob_patterns: list[str] | None = None,
    directories: list[str] | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """Send a query to Gemini CLI and return its response.

    Use this to leverage Gemini's large context window for tasks like:
    - Analyzing large codebases (pass glob patterns or directories to include many files)
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
        directories: Optional list of directory paths to recursively include all files from.
        model: Specific Gemini model to use (no fallback). If not set, tries the fallback chain.
        timeout: Max seconds to wait for Gemini response. Default 120.
    """
    file_paths = await asyncio.to_thread(resolve_files, files, glob_patterns, directories)
    context = await asyncio.to_thread(read_files_as_context, file_paths)

    return await run_gemini(
        prompt=prompt,
        context=context,
        model=model,
        models=None if model else MODEL_FALLBACK_CHAIN,
        timeout=timeout,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gemini_mcp/server.py tests/test_server.py
git commit -m "feat: wire directories parameter into gemini_query tool"
```

---

### Task 3: Extract stats from CLI JSON response

**Files:**
- Modify: `src/gemini_mcp/gemini.py`
- Test: `tests/test_gemini.py`

**Step 1: Write the failing tests**

Add a helper function `_format_metadata` and test it. Also add a test for `_extract_stats`. Add to `tests/test_gemini.py`:

```python
from gemini_mcp.gemini import _format_metadata, _extract_stats


def test_extract_stats_from_json():
    data = {
        "response": "hello",
        "stats": {
            "models": {
                "gemini-3-flash-preview": {
                    "tokens": {"input": 100, "candidates": 50, "total": 150}
                }
            }
        },
    }
    stats = _extract_stats(data)
    assert stats["model"] == "gemini-3-flash-preview"
    assert stats["input_tokens"] == 100
    assert stats["output_tokens"] == 50


def test_extract_stats_multiple_models():
    """When multiple models appear in stats (e.g. routing model + main model),
    pick the one with the most output tokens."""
    data = {
        "response": "hello",
        "stats": {
            "models": {
                "gemini-2.5-flash-lite": {
                    "tokens": {"input": 100, "candidates": 10, "total": 110}
                },
                "gemini-3-flash-preview": {
                    "tokens": {"input": 8000, "candidates": 1333, "total": 9333}
                },
            }
        },
    }
    stats = _extract_stats(data)
    assert stats["model"] == "gemini-3-flash-preview"
    assert stats["output_tokens"] == 1333


def test_extract_stats_empty():
    stats = _extract_stats({"response": "hello", "stats": {}})
    assert stats is None


def test_extract_stats_missing():
    stats = _extract_stats({"response": "hello"})
    assert stats is None


def test_format_metadata_no_fallback():
    stats = {"model": "gemini-3-pro-preview", "input_tokens": 100, "output_tokens": 50}
    result = _format_metadata(stats, fallback_from=None)
    assert result == "---\nModel: gemini-3-pro-preview\nTokens: 100 input / 50 output"


def test_format_metadata_with_fallback():
    stats = {"model": "gemini-3-flash-preview", "input_tokens": 100, "output_tokens": 50}
    result = _format_metadata(stats, fallback_from="gemini-3-pro-preview")
    assert "fallback from gemini-3-pro-preview" in result


def test_format_metadata_none_stats():
    result = _format_metadata(None, fallback_from=None)
    assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gemini.py -v -k "metadata or extract_stats"`
Expected: FAIL — functions don't exist

**Step 3: Implement _extract_stats and _format_metadata**

Add to `src/gemini_mcp/gemini.py`:

```python
def _extract_stats(data: dict) -> dict | None:
    """Extract model and token info from CLI JSON stats."""
    models = data.get("stats", {}).get("models", {})
    if not models:
        return None
    # Pick the model with the most output tokens (skip routing models).
    best_model = max(models, key=lambda m: models[m].get("tokens", {}).get("candidates", 0))
    tokens = models[best_model].get("tokens", {})
    return {
        "model": best_model,
        "input_tokens": tokens.get("input", 0),
        "output_tokens": tokens.get("candidates", 0),
    }


def _format_metadata(stats: dict | None, fallback_from: str | None) -> str | None:
    """Format stats into a metadata footer string."""
    if stats is None:
        return None
    model_line = f"Model: {stats['model']}"
    if fallback_from:
        model_line += f" (fallback from {fallback_from})"
    token_line = f"Tokens: {stats['input_tokens']} input / {stats['output_tokens']} output"
    return f"---\n{model_line}\n{token_line}"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gemini.py -v -k "metadata or extract_stats"`
Expected: All new tests PASS

**Step 5: Commit**

```bash
git add src/gemini_mcp/gemini.py tests/test_gemini.py
git commit -m "feat: add stats extraction and metadata formatting helpers"
```

---

### Task 4: Wire metadata into run_gemini response

**Files:**
- Modify: `src/gemini_mcp/gemini.py`
- Test: `tests/test_gemini.py`

**Step 1: Write the failing tests**

Add to `tests/test_gemini.py`:

```python
@pytest.mark.asyncio
async def test_run_gemini_includes_metadata():
    proc = AsyncMock()
    proc.returncode = 0
    response_json = json.dumps({
        "response": "The answer",
        "stats": {
            "models": {
                "gemini-3-pro-preview": {
                    "tokens": {"input": 500, "candidates": 200, "total": 700}
                }
            }
        },
    })
    proc.communicate = AsyncMock(return_value=(response_json.encode(), b""))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert "The answer" in result
        assert "---" in result
        assert "Model: gemini-3-pro-preview" in result
        assert "500 input / 200 output" in result


@pytest.mark.asyncio
async def test_run_gemini_metadata_shows_fallback():
    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(return_value=(b"", b"model not found"))

    ok_proc = AsyncMock()
    ok_proc.returncode = 0
    ok_json = json.dumps({
        "response": "fallback worked",
        "stats": {
            "models": {
                "gemini-3-flash-preview": {
                    "tokens": {"input": 300, "candidates": 100, "total": 400}
                }
            }
        },
    })
    ok_proc.communicate = AsyncMock(return_value=(ok_json.encode(), b""))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ):
        result = await run_gemini(prompt="hi", models=["bad-model", "gemini-3-flash-preview"])
        assert "fallback worked" in result
        assert "fallback from bad-model" in result
        assert "300 input / 100 output" in result


@pytest.mark.asyncio
async def test_run_gemini_no_metadata_on_plain_text():
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"plain text no json", b""))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == "plain text no json"
        assert "---" not in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gemini.py -v -k "metadata or plain_text"`
Expected: FAIL — `run_gemini` doesn't include metadata yet

**Step 3: Modify _call_gemini and run_gemini to surface metadata**

Change `_call_gemini` to return a 3-tuple `(success, result_text, stats)`:

```python
async def _call_gemini(
    gemini_path: str,
    prompt: str,
    context: str,
    model: str | None,
    timeout: int,
) -> tuple[bool, str, dict | None]:
    """Run Gemini CLI once. Returns (success, result_text, stats)."""
    cmd = [gemini_path, "-p", prompt, "-o", "json"]
    if model:
        cmd.extend(["-m", model])

    stdin_data = context.encode() if context else None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return False, f"Error: gemini CLI timed out after {timeout}s", None
    except OSError as e:
        return False, f"Error: failed to run gemini CLI: {e}", None

    if proc.returncode != 0:
        error_msg = stderr.decode().strip() if stderr else "unknown error"
        return False, f"Error: gemini CLI exited with code {proc.returncode}: {error_msg}", None

    stdout_text = stdout.decode().strip()
    try:
        data = json.loads(stdout_text)
        return True, data.get("response", stdout_text), _extract_stats(data)
    except json.JSONDecodeError:
        for line in reversed(stdout_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return True, data.get("response", stdout_text), _extract_stats(data)
            except (json.JSONDecodeError, AttributeError):
                continue
        return True, stdout_text, None
```

Update `_call_with_retries` to pass through the 3-tuple:

```python
async def _call_with_retries(
    gemini_path: str,
    prompt: str,
    context: str,
    model: str | None,
    timeout: int,
) -> tuple[bool, str, dict | None]:
    """Try a model, retrying on transient errors. Returns (success, result_text, stats)."""
    last_error = ""
    for attempt in range(_MAX_RETRIES + 1):
        success, result, stats = await _call_gemini(gemini_path, prompt, context, model, timeout)
        if success:
            return True, result, stats
        last_error = result
        if _should_fallback(result):
            return False, result, None
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY_SECS)

    return False, last_error, None
```

Update `run_gemini` to append metadata footer:

```python
async def run_gemini(
    prompt: str,
    context: str = "",
    model: str | None = None,
    models: list[str] | None = None,
    timeout: int = 120,
) -> str:
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return "Error: gemini CLI not found in PATH"

    if models:
        to_try = models
    elif model:
        to_try = [model]
    else:
        to_try = [None]

    failures: list[tuple[str, str]] = []
    for m in to_try:
        success, result, stats = await _call_with_retries(
            gemini_path, prompt, context, m, timeout,
        )
        if success:
            parts = []
            if failures:
                parts.append(_format_fallback_warning(failures, m))
            parts.append(result)
            metadata = _format_metadata(
                stats,
                fallback_from=failures[0][0] if failures else None,
            )
            if metadata:
                parts.append(metadata)
            return "\n\n".join(parts)
        failures.append((m or "default", result))

    return failures[-1][1] if failures else "Error: no models to try"
```

**Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS. Some existing tests may need the assertion updated to account for metadata in the response (e.g. `test_run_gemini_simple_prompt` currently asserts exact string match). Update those assertions to use `in` instead of `==` where the response now includes a metadata footer.

Existing tests that need updating:
- `test_run_gemini_simple_prompt`: change `assert result == "This is the response from Gemini."` to `assert "This is the response from Gemini." in result`
- `test_run_gemini_with_context`: same pattern
- `test_run_gemini_handles_warnings_before_json`: same
- `test_run_gemini_handles_braces_in_warnings`: same
- `test_run_gemini_no_warning_when_first_model_succeeds`: same
- `test_rate_limit_retries_same_model`: same

The mock fixture at the top of `test_gemini.py` uses `"stats": {}` which produces no metadata, but `_extract_stats({})` returns `None`, so the footer is omitted. However, these tests should still be updated to use `in` for resilience.

**Step 5: Commit**

```bash
git add src/gemini_mcp/gemini.py tests/test_gemini.py
git commit -m "feat: surface response metadata (model, tokens) from CLI stats"
```

---

### Task 5: Update CLAUDE.md and verify everything

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Update CLAUDE.md architecture section**

Add `directories` to the tool description and mention response metadata:

```markdown
- **server.py** — FastMCP app with one `@mcp.tool`: `gemini_query`. Accepts file paths, glob patterns, and directories. Resolves files, assembles context, delegates to `run_gemini`. Response includes metadata footer (model used, token counts).
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with directory support and response metadata"
```
