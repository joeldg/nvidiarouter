# @spec[GATEWAY_API.md#Requirements]
"""
Auto-inline remote image URLs as base64 data URLs.

NVIDIA's vision NIM requires inline base64 images rather than remote URLs, so
the gateway fetches any http(s) image_url parts and rewrites them.
"""

import base64
from typing import Optional

import structlog

from ..config import settings
from . import runtime

logger = structlog.get_logger()


# @spec[GATEWAY_API.md#Requirements]
async def fetch_as_data_url(url: str) -> Optional[str]:
    """Fetch a remote image and return it as a base64 data URL, or None."""
    try:
        resp = await runtime.http_client.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
        )
        resp.raise_for_status()
        data = resp.content
        if len(data) > settings.image_fetch_max_bytes:
            logger.warning("remote image too large; leaving as URL", bytes=len(data))
            return None
        mime = (resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                or "image/jpeg")
        return f"data:{mime};base64," + base64.b64encode(data).decode()
    except Exception as e:
        logger.warning("failed to inline remote image", url=url, error=str(e))
        return None


# @spec[GATEWAY_API.md#Requirements]
async def inline_remote_images(messages: list) -> list:
    """Replace remote image_url parts with inlined base64 data URLs."""
    if not settings.inline_remote_images:
        return messages
    result = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                ):
                    url = part["image_url"].get("url", "")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        data_url = await fetch_as_data_url(url)
                        if data_url:
                            part = {**part, "image_url": {**part["image_url"], "url": data_url}}
                new_content.append(part)
            msg = {**msg, "content": new_content}
        result.append(msg)
    return result
