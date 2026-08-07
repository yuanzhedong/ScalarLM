"""
Render one /v1/generate prompts-array entry into a prompt string.

The /v1/generate endpoint accepts a batch of independent inference
requests in `prompts: [...]`. Each entry can be one of three shapes:

  - a bare string (legacy raw passthrough),
  - a dict shaped `{"prompt": str, ...}` (raw passthrough, same
    semantics as the bare string),
  - a dict shaped `{"messages": [...], ...}` (chat turns of one
    request, rendered via the model's tokenizer chat template).

This module is the single dispatch point between those forms. The
chat-completions handler renders its own (different-shaped) input
directly via `render_chat_template`; only `/v1/generate` needs the
per-entry shape disambiguation done here.

See docs/openai-chat-completions-queue.md §10 for the wire contract.
"""

from typing import Any

from fastapi import HTTPException

from cray_infra.api.fastapi.chat_completions.render_chat_template import (
    render_chat_template,
)


def render_generate_entry(entry: Any, *, model: str) -> Any:
    if isinstance(entry, str):
        return entry

    if not isinstance(entry, dict):
        raise HTTPException(
            status_code=400,
            detail=(
                "prompts entries must be either a string or a dict with "
                f"`prompt` or `messages`; got {type(entry).__name__}"
            ),
        )

    # Image-bearing entry: {"text": str, "images": [url_or_data_url, ...]}.
    # Passed through the queue as a structured payload instead of a rendered
    # string — the generate worker turns it into OpenAI-format messages and
    # vLLM's chat path applies the model's template (and image processing)
    # exactly once. Rendering here would strip the pixels.
    if entry.get("images"):
        text = entry.get("text") or entry.get("prompt")
        if not text:
            raise HTTPException(
                status_code=400,
                detail="prompts entries with `images` must also carry `text`",
            )
        return {"text": text, "images": entry["images"]}

    try:
        return render_chat_template(
            model=model,
            messages=entry.get("messages"),
            prompt=entry.get("prompt"),
        )
    except ValueError as e:
        # render_chat_template raises ValueError on shape errors
        # (both/neither set, empty values). Surface as 400 so the
        # caller learns about the bad request, not a 500.
        raise HTTPException(status_code=400, detail=str(e))
