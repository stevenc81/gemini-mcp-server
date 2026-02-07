import asyncio
import json
import shutil


async def run_gemini(
    prompt: str,
    context: str = "",
    model: str | None = None,
    timeout: int = 120,
) -> str:
    """Run Gemini CLI in headless mode and return the response text.

    Args:
        prompt: The instruction/question for Gemini.
        context: Optional file context to pipe via stdin.
        model: Optional Gemini model name (e.g., "gemini-2.5-pro").
        timeout: Timeout in seconds. Default 120.
    """
    gemini_path = shutil.which("gemini")
    if not gemini_path:
        return "Error: gemini CLI not found in PATH"

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
        return f"Error: gemini CLI timed out after {timeout}s"
    except OSError as e:
        return f"Error: failed to run gemini CLI: {e}"

    if proc.returncode != 0:
        error_msg = stderr.decode().strip() if stderr else "unknown error"
        return f"Error: gemini CLI exited with code {proc.returncode}: {error_msg}"

    stdout_text = stdout.decode().strip()
    try:
        data = json.loads(stdout_text)
        return data.get("response", stdout_text)
    except json.JSONDecodeError:
        return stdout_text
