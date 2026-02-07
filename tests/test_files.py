import os
import tempfile
from gemini_mcp.files import resolve_files, read_files_as_context


def test_resolve_files_with_explicit_paths():
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(b"hello")
        path = f.name
    try:
        result = resolve_files(files=[path])
        assert result == [path]
    finally:
        os.unlink(path)


def test_resolve_files_with_glob_pattern(tmp_path):
    (tmp_path / "a.py").write_text("aaa")
    (tmp_path / "b.py").write_text("bbb")
    (tmp_path / "c.txt").write_text("ccc")
    result = resolve_files(glob_patterns=[str(tmp_path / "*.py")])
    assert sorted(result) == sorted([str(tmp_path / "a.py"), str(tmp_path / "b.py")])


def test_resolve_files_combines_both(tmp_path):
    explicit = tmp_path / "explicit.py"
    explicit.write_text("explicit")
    (tmp_path / "glob1.py").write_text("glob1")
    result = resolve_files(
        files=[str(explicit)],
        glob_patterns=[str(tmp_path / "glob*.py")],
    )
    assert str(explicit) in result
    assert str(tmp_path / "glob1.py") in result


def test_resolve_files_deduplicates(tmp_path):
    f = tmp_path / "dup.py"
    f.write_text("dup")
    result = resolve_files(
        files=[str(f)],
        glob_patterns=[str(tmp_path / "*.py")],
    )
    assert result.count(str(f)) == 1


def test_resolve_files_skips_missing():
    result = resolve_files(files=["/nonexistent/file.py"])
    assert result == []


def test_read_files_as_context(tmp_path):
    f = tmp_path / "example.py"
    f.write_text("print('hello')")
    context = read_files_as_context([str(f)])
    assert '<file path="' in context
    assert "print('hello')" in context


def test_read_files_as_context_empty():
    context = read_files_as_context([])
    assert context == ""
