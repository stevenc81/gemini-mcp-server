import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from gemini_mcp.gemini import run_gemini, _should_fallback, _format_metadata, _extract_stats


# ---------------------------------------------------------------------------
# Helpers – mock processes for stream-json output
# ---------------------------------------------------------------------------


def _stream_proc(events=None, returncode=0, stderr=b""):
    """Create a mock subprocess that streams events as stream-json lines."""
    proc = MagicMock()

    lines = [json.dumps(e).encode() + b"\n" for e in (events or [])]
    lines.append(b"")  # EOF
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=lines)

    reads = [stderr, b""] if stderr else [b""]
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(side_effect=reads)

    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdin.close = MagicMock()
    proc.stdin.wait_closed = AsyncMock()

    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=returncode)
    proc.kill = MagicMock()
    return proc


def _ok_events(response="This is the response from Gemini.", session_id="test-123", stats=None):
    """Build a standard set of stream-json events for a successful response."""
    return [
        {"type": "init", "session_id": session_id},
        {"type": "message", "role": "assistant", "content": response, "delta": True},
        {"type": "result", "status": "success", "stats": stats or {}},
    ]


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_simple_prompt():
    proc = _stream_proc(_ok_events())
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_gemini(prompt="What is 2+2?")
        assert result == "This is the response from Gemini."
        args = mock_exec.call_args[0]
        assert "gemini" in args[0]
        assert "-p" in args
        assert "-o" in args
        idx = list(args).index("-o")
        assert args[idx + 1] == "stream-json"


@pytest.mark.asyncio
async def test_run_gemini_with_context():
    proc = _stream_proc(_ok_events())
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(
            prompt="Review this code",
            context="<file path=\"test.py\">\nprint('hi')\n</file>",
        )
        assert result == "This is the response from Gemini."
        proc.stdin.write.assert_called_once()
        written = proc.stdin.write.call_args[0][0]
        assert b"test.py" in written


@pytest.mark.asyncio
async def test_run_gemini_with_model():
    proc = _stream_proc(_ok_events())
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await run_gemini(prompt="hello", model="gemini-2.5-pro")
        args = mock_exec.call_args[0]
        assert "-m" in args
        idx = list(args).index("-m")
        assert args[idx + 1] == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_run_gemini_handles_failure():
    proc = _stream_proc(returncode=1, stderr=b"Error: something went wrong")
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="fail")
        assert "Error" in result


@pytest.mark.asyncio
async def test_run_gemini_skips_non_json_lines():
    """Non-JSON lines in stdout (CLI warnings, retries) are silently ignored."""
    events = _ok_events(response="hello")
    lines = [b"Warning: new version available\n"]
    lines += [json.dumps(e).encode() + b"\n" for e in events]
    lines.append(b"")
    proc = _stream_proc()
    proc.stdout.readline = AsyncMock(side_effect=lines)
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == "hello"


@pytest.mark.asyncio
async def test_run_gemini_assembles_multiple_deltas():
    """Multiple assistant message deltas are concatenated."""
    events = [
        {"type": "init", "session_id": "x"},
        {"type": "message", "role": "assistant", "content": "Hello ", "delta": True},
        {"type": "message", "role": "assistant", "content": "world!", "delta": True},
        {"type": "result", "status": "success", "stats": {}},
    ]
    proc = _stream_proc(events)
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == "Hello world!"


@pytest.mark.asyncio
async def test_run_gemini_empty_response_on_no_events():
    """If stdout has no JSON events, response is empty."""
    proc = _stream_proc()  # no events
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == ""


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_fallback_chain():
    """First model fails, second succeeds — response includes warning."""
    fail_proc = _stream_proc(returncode=1, stderr=b"model not found")
    ok_proc = _stream_proc(_ok_events(response="fallback worked"))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ) as mock_exec:
        result = await run_gemini(prompt="hi", models=["bad-model", "good-model"])
        assert "fallback worked" in result
        assert "[WARNING: Fell back to good-model]" in result
        assert "bad-model" in result
        assert mock_exec.call_count == 2


@pytest.mark.asyncio
async def test_run_gemini_no_warning_when_first_model_succeeds():
    """No warning when the first model works fine."""
    proc = _stream_proc(_ok_events(response="first try"))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi", models=["model-a", "model-b"])
        assert result == "first try"
        assert "WARNING" not in result


@pytest.mark.asyncio
async def test_run_gemini_fallback_all_fail():
    """All models fail, returns last error."""
    procs = [_stream_proc(returncode=1, stderr=b"model unavailable") for _ in range(2)]
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", side_effect=procs):
        result = await run_gemini(prompt="hi", models=["bad1", "bad2"])
        assert "Error" in result


# ---------------------------------------------------------------------------
# _should_fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "Error: model xyz not found",
    "model does not exist",
    "model unavailable",
    "HTTP 404 response",
    "model not supported for this request",
    "this model is deprecated",
])
def test_should_fallback_on_availability_errors(msg):
    assert _should_fallback(msg) is True


@pytest.mark.parametrize("msg", [
    "Error: 429 Too Many Requests",
    "rate limit exceeded",
    "quota exceeded for project",
    "RESOURCE_EXHAUSTED",
    "service overloaded, try again",
    "too many requests",
    "503 Service Temporarily Unavailable",
    "temporarily unavailable",
])
def test_should_not_fallback_on_transient_errors(msg):
    assert _should_fallback(msg) is False


def test_should_fallback_on_unknown_errors():
    """Unknown errors default to fallback."""
    assert _should_fallback("Error: something totally unexpected") is True


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_retries_same_model():
    """Rate limit error retries the same model instead of falling back."""
    fail_proc = _stream_proc(returncode=1, stderr=b"429 rate limit exceeded")
    ok_proc = _stream_proc(_ok_events(response="retry worked"))

    with (
        patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", side_effect=[fail_proc, ok_proc]) as mock_exec,
        patch("gemini_mcp.gemini.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await run_gemini(prompt="hi", model="gemini-3-pro-preview")
        assert result == "retry worked"
        assert mock_exec.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausts_retries_then_falls_back():
    """If retries are exhausted on a transient error, fall back to the
    next model but include a warning."""
    fail_procs = [_stream_proc(returncode=1, stderr=b"429 rate limit exceeded") for _ in range(3)]
    ok_proc = _stream_proc(_ok_events(response="flash answered"))

    with (
        patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", side_effect=[*fail_procs, ok_proc]) as mock_exec,
        patch("gemini_mcp.gemini.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await run_gemini(
            prompt="hi",
            models=["gemini-3-pro-preview", "gemini-2.5-flash"],
        )
        assert "flash answered" in result
        assert "[WARNING: Fell back to gemini-2.5-flash]" in result
        assert "gemini-3-pro-preview" in result
        # 3 retries on first model + 1 success on second = 4
        assert mock_exec.call_count == 4


@pytest.mark.asyncio
async def test_model_not_found_skips_retries():
    """A 'model not found' error falls back immediately without retrying."""
    fail_proc = _stream_proc(returncode=1, stderr=b"model not found")
    ok_proc = _stream_proc(_ok_events(response="fallback"))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ) as mock_exec:
        result = await run_gemini(prompt="hi", models=["bad-model", "good-model"])
        assert "fallback" in result
        assert "[WARNING: Fell back to good-model]" in result
        assert mock_exec.call_count == 2


# ---------------------------------------------------------------------------
# _extract_stats tests
# ---------------------------------------------------------------------------


def test_extract_stats_from_json():
    """Original -o json format with nested tokens dict."""
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


def test_extract_stats_stream_json_format():
    """stream-json format with flat token counts."""
    data = {
        "session_id": "abc",
        "stats": {
            "models": {
                "gemini-2.5-flash": {
                    "total_tokens": 5002,
                    "input_tokens": 4955,
                    "output_tokens": 1,
                    "cached": 0,
                }
            }
        },
    }
    stats = _extract_stats(data)
    assert stats["model"] == "gemini-2.5-flash"
    assert stats["input_tokens"] == 4955
    assert stats["output_tokens"] == 1
    assert stats["session_id"] == "abc"


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


# ---------------------------------------------------------------------------
# _format_metadata tests
# ---------------------------------------------------------------------------


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


def test_format_metadata_with_skipped_files():
    stats = {"model": "gemini-3-pro-preview", "input_tokens": 100, "output_tokens": 50}
    result = _format_metadata(stats, fallback_from=None, skipped_files=14)
    assert "Skipped: 14 binary/junk files" in result
    assert "Model: gemini-3-pro-preview" in result


def test_format_metadata_no_skipped_files():
    stats = {"model": "gemini-3-pro-preview", "input_tokens": 100, "output_tokens": 50}
    result = _format_metadata(stats, fallback_from=None, skipped_files=0)
    assert "Skipped" not in result


def test_format_metadata_with_session_id():
    stats = {
        "model": "gemini-3-pro-preview",
        "input_tokens": 100,
        "output_tokens": 50,
        "session_id": "abc-123",
    }
    result = _format_metadata(stats, fallback_from=None)
    assert "Session ID: abc-123" in result


def test_format_metadata_without_session_id():
    stats = {"model": "gemini-3-pro-preview", "input_tokens": 100, "output_tokens": 50}
    result = _format_metadata(stats, fallback_from=None)
    assert "Session ID" not in result


# ---------------------------------------------------------------------------
# Metadata wiring (end-to-end through run_gemini)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_includes_metadata():
    events = _ok_events(
        response="The answer",
        session_id="s1",
        stats={
            "models": {
                "gemini-3-pro-preview": {
                    "input_tokens": 500, "output_tokens": 200,
                }
            }
        },
    )
    proc = _stream_proc(events)
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert "The answer" in result
        assert "---" in result
        assert "Model: gemini-3-pro-preview" in result
        assert "500 input / 200 output" in result


@pytest.mark.asyncio
async def test_run_gemini_metadata_shows_fallback():
    fail_proc = _stream_proc(returncode=1, stderr=b"model not found")
    ok_events = _ok_events(
        response="fallback worked",
        stats={
            "models": {
                "gemini-3-flash-preview": {
                    "input_tokens": 300, "output_tokens": 100,
                }
            }
        },
    )
    ok_proc = _stream_proc(ok_events)

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ):
        result = await run_gemini(prompt="hi", models=["bad-model", "gemini-3-flash-preview"])
        assert "fallback worked" in result
        assert "fallback from bad-model" in result
        assert "300 input / 100 output" in result


@pytest.mark.asyncio
async def test_run_gemini_no_metadata_without_result_event():
    """No metadata when there's no result event with stats."""
    events = [
        {"type": "init", "session_id": "x"},
        {"type": "message", "role": "assistant", "content": "hello", "delta": True},
    ]
    proc = _stream_proc(events)
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == "hello"
        assert "---" not in result


def test_extract_stats_includes_session_id():
    data = {
        "session_id": "abc-123",
        "response": "hello",
        "stats": {
            "models": {
                "gemini-3-pro-preview": {
                    "tokens": {"input": 100, "candidates": 50, "total": 150}
                }
            }
        },
    }
    stats = _extract_stats(data)
    assert stats["session_id"] == "abc-123"


def test_extract_stats_no_session_id():
    data = {
        "response": "hello",
        "stats": {
            "models": {
                "gemini-3-pro-preview": {
                    "tokens": {"input": 100, "candidates": 50, "total": 150}
                }
            }
        },
    }
    stats = _extract_stats(data)
    assert stats.get("session_id") is None


# ---------------------------------------------------------------------------
# Session continuity (--resume flag)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_passes_resume_flag():
    """When session_id is provided, CLI should get -r flag."""
    proc = _stream_proc(_ok_events(response="resumed", session_id="abc-123"))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        result = await run_gemini(prompt="follow up", session_id="abc-123")
        call_args = mock_exec.call_args[0]
        assert "-r" in call_args
        idx = list(call_args).index("-r")
        assert call_args[idx + 1] == "abc-123"


@pytest.mark.asyncio
async def test_run_gemini_no_resume_without_session_id():
    """Without session_id, no -r flag."""
    proc = _stream_proc(_ok_events(response="fresh", session_id="new-session"))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        await run_gemini(prompt="hello")
        call_args = mock_exec.call_args[0]
        assert "-r" not in call_args


@pytest.mark.asyncio
async def test_run_gemini_drops_resume_on_fallback():
    """When falling back to a different model, drop --resume."""
    fail_proc = _stream_proc(returncode=1, stderr=b"model not found")
    ok_proc = _stream_proc(_ok_events(response="fallback", session_id="new-id"))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ) as mock_exec:
        await run_gemini(
            prompt="hi",
            models=["bad-model", "good-model"],
            session_id="old-session",
        )
        first_call_args = mock_exec.call_args_list[0][0]
        second_call_args = mock_exec.call_args_list[1][0]
        assert "-r" in first_call_args
        assert "-r" not in second_call_args


# ---------------------------------------------------------------------------
# Inactivity timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_kills_on_streaming_stall():
    """Process is killed when streaming starts then goes silent."""
    # Emit init + one assistant delta, then hang (enters streaming phase)
    lines = [
        json.dumps({"type": "init", "session_id": "x"}).encode() + b"\n",
        json.dumps({"type": "message", "role": "assistant", "content": "partial", "delta": True}).encode() + b"\n",
    ]
    call_count = {"n": 0}

    async def readline_then_hang(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= len(lines):
            return lines[call_count["n"] - 1]
        await asyncio.Future()  # hang forever — streaming stall

    proc = MagicMock()
    proc.returncode = -9
    proc.stdout = MagicMock()
    proc.stdout.readline = readline_then_hang
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(side_effect=[b""])
    proc.stdin = None
    proc.wait = AsyncMock(return_value=-9)
    proc.kill = MagicMock()

    with (
        patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc),
        patch("gemini_mcp.gemini._STREAMING_TIMEOUT", 0.1),
    ):
        result = await run_gemini(prompt="hang", timeout=30)
        assert "stalled" in result
        proc.kill.assert_called()


@pytest.mark.asyncio
async def test_run_gemini_hard_timeout():
    """Hard wall-clock timeout fires when process streams slowly but exceeds limit."""
    call_count = {"n": 0}

    async def slow_readline(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return json.dumps({"type": "init", "session_id": "x"}).encode() + b"\n"
        # Keep producing output to prevent inactivity timeout, but slowly
        await asyncio.sleep(0.05)
        return json.dumps({"type": "message", "role": "assistant", "content": ".", "delta": True}).encode() + b"\n"

    proc = MagicMock()
    proc.returncode = None
    proc.stdout = MagicMock()
    proc.stdout.readline = slow_readline
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(side_effect=[b""])
    proc.stdin = None
    proc.wait = AsyncMock(return_value=-9)
    proc.kill = MagicMock()

    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="slow", timeout=1)
        assert "killed" in result or "timed out" in result or "signal" in result
        proc.kill.assert_called()


@pytest.mark.asyncio
async def test_run_gemini_process_dies_mid_stream():
    """If process exits with error after partial output, error is returned."""
    events = [
        {"type": "init", "session_id": "x"},
        {"type": "message", "role": "assistant", "content": "partial ", "delta": True},
    ]
    # Process crashes after emitting partial response (returncode 1)
    proc = _stream_proc(events, returncode=1, stderr=b"internal error")
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="crash")
        assert "Error" in result
        assert "internal error" in result
        # Partial response should NOT be returned
        assert "partial" not in result


@pytest.mark.asyncio
async def test_run_gemini_stdin_write_failure():
    """Broken pipe on stdin write is handled gracefully."""
    events = _ok_events(response="still works")
    proc = _stream_proc(events)
    proc.stdin.drain = AsyncMock(side_effect=BrokenPipeError("pipe closed"))
    proc.stdin.wait_closed = AsyncMock()

    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        # Should not crash — process may have responded despite stdin failure
        result = await run_gemini(prompt="hi", context="some context")
        assert result == "still works"
