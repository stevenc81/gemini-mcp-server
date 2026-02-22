import pytest
from unittest.mock import patch, AsyncMock
from gemini_mcp.server import gemini_query, MODEL_FALLBACK_CHAIN


@pytest.mark.asyncio
async def test_gemini_query_prompt_only():
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "The answer is 4"
        result = await gemini_query.fn(prompt="What is 2+2?")
        assert result == "The answer is 4"
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["model"] is None
        assert call_kwargs["models"] == MODEL_FALLBACK_CHAIN


@pytest.mark.asyncio
async def test_gemini_query_with_files(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1")
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Looks good"
        result = await gemini_query.fn(
            prompt="Review this",
            files=[str(f)],
        )
        assert result == "Looks good"
        call_kwargs = mock_run.call_args[1]
        assert "code.py" in call_kwargs["context"]
        assert "x = 1" in call_kwargs["context"]


@pytest.mark.asyncio
async def test_gemini_query_with_glob(tmp_path):
    (tmp_path / "a.py").write_text("aaa")
    (tmp_path / "b.py").write_text("bbb")
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Found patterns"
        result = await gemini_query.fn(
            prompt="Analyze",
            glob_patterns=[str(tmp_path / "*.py")],
        )
        assert result == "Found patterns"
        call_kwargs = mock_run.call_args[1]
        assert "aaa" in call_kwargs["context"]
        assert "bbb" in call_kwargs["context"]


@pytest.mark.asyncio
async def test_gemini_query_with_explicit_model():
    """When model is explicitly set, use it directly with no fallback chain."""
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "ok"
        await gemini_query.fn(prompt="hi", model="gemini-2.5-pro")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"
        assert call_kwargs["models"] is None


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


@pytest.mark.asyncio
async def test_gemini_query_passes_skipped_count(tmp_path):
    (tmp_path / "app.py").write_text("code")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "result"
        await gemini_query.fn(prompt="Review", directories=[str(tmp_path)])
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["skipped_files"] == 1


@pytest.mark.asyncio
async def test_gemini_query_no_files_no_globs():
    with patch("gemini_mcp.server.run_gemini", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "response"
        await gemini_query.fn(prompt="hello")
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["context"] == ""
