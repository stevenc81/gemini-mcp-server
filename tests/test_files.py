import os
import tempfile
from gemini_mcp.files import resolve_files, read_files_as_context, _should_skip_file


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


def test_resolve_files_respects_max_files(tmp_path):
    for i in range(10):
        (tmp_path / f"file{i}.py").write_text(f"content{i}")
    result = resolve_files(glob_patterns=[str(tmp_path / "*.py")], max_files=3)
    assert len(result) == 3


def test_read_files_as_context_respects_max_bytes(tmp_path):
    paths = []
    for i in range(3):
        f = tmp_path / f"file{i}.txt"
        f.write_text("x" * 100)
        paths.append(str(f))
    context = read_files_as_context(paths, max_bytes=150)
    assert "Context truncated" in context
    assert "byte limit" in context
    # First file should be included, not all three
    assert 'file0.txt' in context
    assert 'file2.txt' not in context


def test_read_files_as_context_skips_oversized_first_file(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 200)
    context = read_files_as_context([str(big)], max_bytes=100)
    assert "big.txt" not in context or 'error=' in context or "Context truncated" in context
    assert "Context truncated" in context
    assert "0 of 1 files included" in context


def test_resolve_files_with_directories(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.py").write_text("root")
    (sub / "nested.py").write_text("nested")
    result = resolve_files(directories=[str(tmp_path)])
    basenames = [os.path.basename(p) for p in result]
    assert "root.py" in basenames
    assert "nested.py" in basenames


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


def test_resolve_files_directories_skips_junk_dirs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("app")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("gitconfig")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "mod.cpython-312.pyc").write_text("bytecode")
    node = tmp_path / "node_modules"
    node.mkdir()
    (node / "pkg.js").write_text("pkg")
    result = resolve_files(directories=[str(tmp_path)])
    basenames = [os.path.basename(p) for p in result]
    assert "app.py" in basenames
    assert "config" not in basenames
    assert "mod.cpython-312.pyc" not in basenames
    assert "pkg.js" not in basenames


def test_should_skip_binary_extensions():
    assert _should_skip_file("photo.png") is True
    assert _should_skip_file("image.jpg") is True
    assert _should_skip_file("font.woff2") is True
    assert _should_skip_file("lib.so") is True
    assert _should_skip_file("archive.zip") is True
    assert _should_skip_file("bundle.map") is True


def test_should_skip_junk_files():
    assert _should_skip_file(".DS_Store") is True


def test_should_not_skip_text_files():
    assert _should_skip_file("app.py") is False
    assert _should_skip_file("README.md") is False
    assert _should_skip_file("package.json") is False
    assert _should_skip_file("yarn.lock") is False


def test_should_skip_case_insensitive():
    assert _should_skip_file("IMAGE.PNG") is True
    assert _should_skip_file("Photo.JPG") is True
