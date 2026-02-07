import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from gemini_mcp.gemini import run_gemini


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
    """First model fails, second succeeds."""
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
        assert result == "fallback worked"
        assert mock_exec.call_count == 2


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
