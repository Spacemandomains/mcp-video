"""Optional Meta Model API analysis for mcp-video.

mcp-video's core tools return frames + transcript to the *host* model and
make no LLM call of their own. This module adds an opt-in direct call to
the Meta Model API (dev.meta.ai) for hosts without vision:

    base URL : https://api.meta.ai/v1   (META_BASE_URL override)
    model    : muse-spark-1.3           (META_MODEL override, multimodal)
    key      : MODEL_API_KEY env var only — never hard-coded

Uses only the standard library (urllib), so no new dependency is added.
"""

from __future__ import annotations

import json
import os
import urllib.request

META_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-spark-1.3"
META_KEY_ENV = "MODEL_API_KEY"


def meta_config() -> dict:
    """Resolve Meta config from the environment (no key value is logged)."""
    return {
        "base_url": (os.environ.get("META_BASE_URL") or META_BASE_URL).rstrip("/"),
        "model": os.environ.get("META_MODEL") or META_DEFAULT_MODEL,
        "api_key": os.environ.get(META_KEY_ENV) or "",
        "key_env": META_KEY_ENV,
    }


def analyze_frames_with_meta(
    frames: list[dict],
    prompt: str,
    transcript: str = "",
) -> str:
    """Send extracted frames (base64 JPEGs, as produced by video.py) plus an
    optional transcript to Muse Spark via Chat Completions and return the
    model's text analysis. Raises RuntimeError if MODEL_API_KEY is unset."""
    cfg = meta_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            f"{META_KEY_ENV} is not set. Create a Meta Model API key at "
            "https://dev.meta.ai/ and export it as MODEL_API_KEY."
        )

    content: list[dict] = [{"type": "text", "text": prompt}]
    if transcript:
        content.append(
            {"type": "text", "text": f"Transcript:\n{transcript[:5000]}"}
        )
    for frame in frames[:50]:  # Meta limit: 50 images per request
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['image_base64']}"
                },
            }
        )

    payload = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        f"{cfg['base_url']}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]
