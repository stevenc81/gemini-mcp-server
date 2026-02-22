import asyncio
import json
import re
import shutil

# Patterns indicating the model itself is unavailable or doesn't exist.
# These warrant falling back to the next model.
_FALLBACK_PATTERNS = [
    re.compile(r"model.*not found", re.IGNORECASE),
    re.compile(r"model.*does not exist", re.IGNORECASE),
    re.compile(r"model.*unavailable", re.IGNORECASE),
    re.compile(r"(?:^|[\s:])404(?:\s|$)"),
    re.compile(r"not supported", re.IGNORECASE),
    re.compile(r"deprecated", re.IGNORECASE),
]

# Patterns indicating a transient error that could succeed on retry
# with the same model. Don't fall back for these.
_RETRIABLE_PATTERNS = [
    re.compile(r"(?:^|[\s:])429(?:\s|$)"),
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"quota", re.IGNORECASE),
    re.compile(r"resource.?exhausted", re.IGNORECASE),
    re.compile(r"overloaded", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"(?:^|[\s:])503(?:\s|$)"),
    re.compile(r"temporarily unavailable", re.IGNORECASE),
]

_MAX_RETRIES = 2
_RETRY_DELAY_SECS = 3


def _should_fallback(error_msg: str) -> bool:
    """Return True if the error means this model can't serve the request."""
    for pat in _RETRIABLE_PATTERNS:
        if pat.search(error_msg):
            return False
    for pat in _FALLBACK_PATTERNS:
        if pat.search(error_msg):
            return True
    # Unknown error — fall back rather than retrying blindly.
    return True


def _extract_stats(data: dict) -> dict | None:
    """Extract model, token info, and session ID from CLI JSON."""
    models = data.get("stats", {}).get("models", {})
    if not models:
        return None
    # Pick the model with the most output tokens (skip routing models).
    best_model = max(models, key=lambda m: models[m].get("tokens", {}).get("candidates", 0))
    tokens = models[best_model].get("tokens", {})
    result = {
        "model": best_model,
        "input_tokens": tokens.get("input", 0),
        "output_tokens": tokens.get("candidates", 0),
    }
    session_id = data.get("session_id")
    if session_id:
        result["session_id"] = session_id
    return result


def _format_metadata(stats: dict | None, fallback_from: str | None, skipped_files: int = 0) -> str | None:
    """Format stats into a metadata footer string."""
    if stats is None:
        return None
    model_line = f"Model: {stats['model']}"
    if fallback_from:
        model_line += f" (fallback from {fallback_from})"
    token_line = f"Tokens: {stats['input_tokens']} input / {stats['output_tokens']} output"
    lines = ["---", model_line, token_line]
    if stats.get("session_id"):
        lines.append(f"Session ID: {stats['session_id']}")
    if skipped_files > 0:
        lines.append(f"Skipped: {skipped_files} binary/junk files")
    return "\n".join(lines)


async def _call_gemini(
    gemini_path: str,
    prompt: str,
    context: str,
    model: str | None,
    timeout: int,
    session_id: str | None = None,
) -> tuple[bool, str, dict | None]:
    """Run Gemini CLI once. Returns (success, result_text, stats_or_none)."""
    cmd = [gemini_path, "-p", prompt, "-o", "json"]
    if model:
        cmd.extend(["-m", model])
    if session_id:
        cmd.extend(["-r", session_id])

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


async def _call_with_retries(
    gemini_path: str,
    prompt: str,
    context: str,
    model: str | None,
    timeout: int,
    session_id: str | None = None,
) -> tuple[bool, str, dict | None]:
    """Try a model, retrying on transient errors.

    Returns (success, result_text, stats_or_none).
    """
    last_error = ""
    for attempt in range(_MAX_RETRIES + 1):
        success, result, stats = await _call_gemini(gemini_path, prompt, context, model, timeout, session_id)
        if success:
            return True, result, stats
        last_error = result
        if _should_fallback(result):
            return False, result, None
        # Transient error — retry after a delay
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY_SECS)

    # Exhausted retries on a transient error — fall back, but caller
    # will record that we tried this model.
    return False, last_error, None


async def run_gemini(
    prompt: str,
    context: str = "",
    model: str | None = None,
    models: list[str] | None = None,
    timeout: int = 120,
    skipped_files: int = 0,
    session_id: str | None = None,
) -> str:
    """Run Gemini CLI in headless mode and return the response text.

    Tries each model in order until one succeeds. Transient errors
    (rate limits, 503s) are retried on the same model before falling
    back. Permanent errors (404, model not found) fall back immediately.

    When a fallback occurs, the response is prefixed with a warning
    showing which models failed and why.

    If a single model is provided via `model`, it is used directly with
    no fallback (but transient errors are still retried).

    Args:
        prompt: The instruction/question for Gemini.
        context: Optional file context to pipe via stdin.
        model: Single model to use (no fallback).
        models: Ordered list of models to try. First success wins.
        timeout: Timeout in seconds. Default 120.
        session_id: Optional session ID to resume a previous conversation.
    """
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return "Error: gemini CLI not found in PATH"

    if models:
        to_try = models
    elif model:
        to_try = [model]
    else:
        to_try = [None]

    failures: list[tuple[str, str]] = []  # (model_name, error_msg)
    current_session_id = session_id
    for m in to_try:
        success, result, stats = await _call_with_retries(
            gemini_path, prompt, context, m, timeout, current_session_id,
        )
        if success:
            parts = []
            if failures:
                parts.append(_format_fallback_warning(failures, m))
            parts.append(result)
            metadata = _format_metadata(
                stats,
                fallback_from=failures[0][0] if failures else None,
                skipped_files=skipped_files,
            )
            if metadata:
                parts.append(metadata)
            return "\n\n".join(parts)
        failures.append((m or "default", result))
        current_session_id = None  # Drop resume on fallback

    return failures[-1][1] if failures else "Error: no models to try"


def _format_fallback_warning(
    failures: list[tuple[str, str]],
    used_model: str | None,
) -> str:
    """Build a warning string describing which models failed."""
    lines = [f"[WARNING: Fell back to {used_model or 'default'}]"]
    for model_name, error in failures:
        # Extract just the error detail, not the full "Error: gemini CLI exited..." prefix.
        short = error.split(": ", 2)[-1] if ": " in error else error
        lines.append(f"  - {model_name}: {short}")
    return "\n".join(lines)
