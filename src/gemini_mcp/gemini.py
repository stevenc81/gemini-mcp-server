import asyncio
import json
import shutil


async def _call_gemini(
    gemini_path: str,
    prompt: str,
    context: str,
    model: str | None,
    timeout: int,
) -> tuple[bool, str]:
    """Run Gemini CLI once. Returns (success, result_text)."""
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
        return False, f"Error: gemini CLI timed out after {timeout}s"
    except OSError as e:
        return False, f"Error: failed to run gemini CLI: {e}"

    if proc.returncode != 0:
        error_msg = stderr.decode().strip() if stderr else "unknown error"
        return False, f"Error: gemini CLI exited with code {proc.returncode}: {error_msg}"

    stdout_text = stdout.decode().strip()
    try:
        data = json.loads(stdout_text)
        return True, data.get("response", stdout_text)
    except json.JSONDecodeError:
        for line in reversed(stdout_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                return True, data.get("response", stdout_text)
            except (json.JSONDecodeError, AttributeError):
                continue
        return True, stdout_text


async def run_gemini(
    prompt: str,
    context: str = "",
    model: str | None = None,
    models: list[str] | None = None,
    timeout: int = 120,
) -> str:
    """Run Gemini CLI in headless mode and return the response text.

    Tries each model in order until one succeeds. If a single model is
    provided via `model`, it is used directly with no fallback.

    Args:
        prompt: The instruction/question for Gemini.
        context: Optional file context to pipe via stdin.
        model: Single model to use (no fallback).
        models: Ordered list of models to try. First success wins.
        timeout: Timeout in seconds. Default 120.
    """
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return "Error: gemini CLI not found in PATH"

    # Build the list of models to try
    if models:
        to_try = models
    elif model:
        to_try = [model]
    else:
        to_try = [None]

    last_error = ""
    for m in to_try:
        success, result = await _call_gemini(gemini_path, prompt, context, m, timeout)
        if success:
            return result
        last_error = result

    return last_error
