# @spec[GATEWAY_API.md#Requirements]
"""
Shared gateway runtime state.

Holds the process-wide async HTTP client used to talk to NVIDIA NIM. It's a
module attribute (not a ``from`` import) so the lifespan handler can swap it in
at startup and every module sees the update via ``runtime.http_client``.
"""

from typing import Optional

import httpx

# Initialised by the app lifespan handler; None until the gateway starts.
http_client: Optional[httpx.AsyncClient] = None
