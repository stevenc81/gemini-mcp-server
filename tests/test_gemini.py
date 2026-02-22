import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from gemini_mcp.gemini import run_gemini, _should_fallback, _format_metadata, _extract_stats


@pytest.fixture
def mock_process():
    """Create a mock subprocess that returns a successful Gemini response."""
    proc = AsyncMock()
    proc.returncode = 0
    response_json = json.dumps({
        "session_id": "test-123",
        "response": "This is the response from Gemini.",
        "stats": {},
    })
    proc.communicate = AsyncMock(return_value=(response_json.encode(), b""))
    return proc


@pytest.mark.asyncio
async def test_run_gemini_simple_prompt(mock_process):
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await run_gemini(prompt="What is 2+2?")
        assert result == "This is the response from Gemini."
        call_args = mock_exec.call_args
        args = call_args[0]
        assert "gemini" in args[0]
        assert "-p" in args


@pytest.mark.asyncio
async def test_run_gemini_with_context(mock_process):
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        result = await run_gemini(
            prompt="Review this code",
            context="<file path=\"test.py\">\nprint('hi')\n</file>",
        )
        assert result == "This is the response from Gemini."
        stdin_data = mock_process.communicate.call_args[1].get("input") or mock_process.communicate.call_args[0][0]
        assert b"test.py" in stdin_data


@pytest.mark.asyncio
async def test_run_gemini_with_model(mock_process):
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        await run_gemini(prompt="hello", model="gemini-2.5-pro")
        call_args = mock_exec.call_args[0]
        assert "-m" in call_args
        idx = list(call_args).index("-m")
        assert call_args[idx + 1] == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_run_gemini_handles_failure():
    proc = AsyncMock()
    proc.returncode = 1
    proc.communicate = AsyncMock(return_value=(b"", b"Error: something went wrong"))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="fail")
        assert "Error" in result


@pytest.mark.asyncio
async def test_run_gemini_handles_warnings_before_json():
    proc = AsyncMock()
    proc.returncode = 0
    stdout = b'Warning: new version available\n{"session_id":"x","response":"hello","stats":{}}'
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == "hello"


@pytest.mark.asyncio
async def test_run_gemini_handles_braces_in_warnings():
    proc = AsyncMock()
    proc.returncode = 0
    stdout = b'Warning: check {config} file\n{"session_id":"x","response":"hello","stats":{}}'
    proc.communicate = AsyncMock(return_value=(stdout, b""))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="hi")
        assert result == "hello"


@pytest.mark.asyncio
async def test_run_gemini_handles_invalid_json():
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(b"not json at all", b""))
    with patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", return_value=proc):
        result = await run_gemini(prompt="bad json")
        assert result == "not json at all"


@pytest.mark.asyncio
async def test_run_gemini_fallback_chain():
    """First model fails, second succeeds — response includes warning."""
    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(return_value=(b"", b"model not found"))

    ok_proc = AsyncMock()
    ok_proc.returncode = 0
    ok_json = json.dumps({"session_id": "x", "response": "fallback worked", "stats": {}})
    ok_proc.communicate = AsyncMock(return_value=(ok_json.encode(), b""))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ) as mock_exec:
        result = await run_gemini(
            prompt="hi",
            models=["bad-model", "good-model"],
        )
        assert "fallback worked" in result
        assert "[WARNING: Fell back to good-model]" in result
        assert "bad-model" in result
        assert mock_exec.call_count == 2


@pytest.mark.asyncio
async def test_run_gemini_no_warning_when_first_model_succeeds():
    """No warning when the first model works fine."""
    ok_proc = AsyncMock()
    ok_proc.returncode = 0
    ok_json = json.dumps({"session_id": "x", "response": "first try", "stats": {}})
    ok_proc.communicate = AsyncMock(return_value=(ok_json.encode(), b""))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        return_value=ok_proc,
    ):
        result = await run_gemini(
            prompt="hi",
            models=["model-a", "model-b"],
        )
        assert result == "first try"
        assert "WARNING" not in result


@pytest.mark.asyncio
async def test_run_gemini_fallback_all_fail():
    """All models fail, returns last error."""
    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(return_value=(b"", b"model unavailable"))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        return_value=fail_proc,
    ):
        result = await run_gemini(
            prompt="hi",
            models=["bad1", "bad2"],
        )
        assert "Error" in result


# --- _should_fallback tests ---


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


# --- Retry behavior tests ---


@pytest.mark.asyncio
async def test_rate_limit_retries_same_model():
    """Rate limit error retries the same model instead of falling back."""
    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(return_value=(b"", b"429 rate limit exceeded"))

    ok_proc = AsyncMock()
    ok_proc.returncode = 0
    ok_json = json.dumps({"session_id": "x", "response": "retry worked", "stats": {}})
    ok_proc.communicate = AsyncMock(return_value=(ok_json.encode(), b""))

    with (
        patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", side_effect=[fail_proc, ok_proc]) as mock_exec,
        patch("gemini_mcp.gemini.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await run_gemini(prompt="hi", model="gemini-3-pro-preview")
        assert result == "retry worked"
        # Both calls should use the same model — no fallback.
        assert mock_exec.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausts_retries_then_falls_back():
    """If retries are exhausted on a transient error, fall back to the
    next model but include a warning."""
    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(return_value=(b"", b"429 rate limit exceeded"))

    ok_proc = AsyncMock()
    ok_proc.returncode = 0
    ok_json = json.dumps({"session_id": "x", "response": "flash answered", "stats": {}})
    ok_proc.communicate = AsyncMock(return_value=(ok_json.encode(), b""))

    with (
        patch("gemini_mcp.gemini.asyncio.create_subprocess_exec", side_effect=[fail_proc, fail_proc, fail_proc, ok_proc]) as mock_exec,
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
    fail_proc = AsyncMock()
    fail_proc.returncode = 1
    fail_proc.communicate = AsyncMock(return_value=(b"", b"model not found"))

    ok_proc = AsyncMock()
    ok_proc.returncode = 0
    ok_json = json.dumps({"session_id": "x", "response": "fallback", "stats": {}})
    ok_proc.communicate = AsyncMock(return_value=(ok_json.encode(), b""))

    with patch(
        "gemini_mcp.gemini.asyncio.create_subprocess_exec",
        side_effect=[fail_proc, ok_proc],
    ) as mock_exec:
        result = await run_gemini(
            prompt="hi",
            models=["bad-model", "good-model"],
        )
        assert "fallback" in result
        assert "[WARNING: Fell back to good-model]" in result
        # Exactly 2 calls: one fail (no retry), one success.
        assert mock_exec.call_count == 2


# --- _extract_stats tests ---


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


# --- _format_metadata tests ---


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
