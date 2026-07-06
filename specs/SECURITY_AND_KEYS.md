# Security and Keys

## Scope
Handling of NVIDIA API keys and inbound gateway access: the multi-key pool and
rotation, per-key budgeting, key masking, `.env` handling, optional inbound
client API-key auth, inbound rate limiting, and CORS. Project-scoped; narrows
the global `GLOBAL_SECURITY.md` and `SECURITY_AND_SECRETS.md` for this repo.

## Intent
Upstream credentials must never leak, throughput must scale across multiple keys
without exceeding per-key limits, and operators must be able to require inbound
authentication without breaking local use.

## Requirements
1. NVIDIA API keys MUST be loaded only from the environment/`.env`
   (`NVIDIA_API_KEY`, `NVIDIA_API_KEYS`), MUST NOT be committed, and `.env` MUST
   be gitignored. (Narrows GLOBAL_SECURITY req.1.)
2. Keys MUST be masked wherever surfaced (logs, `/metrics`, diagnostics); full
   key values MUST NOT be printed. (Narrows SECURITY_AND_SECRETS req.1, req.4.)
3. The key pool MUST track per-key usage in a rolling window, select the key
   with the most remaining budget, and cool a key down on upstream 429, failing
   over to another key; when all keys are exhausted the gateway MUST return 503
   with `Retry-After`.
4. Upstream requests MUST send `Authorization: Bearer <key>` over the configured
   HTTPS NIM base URL.
5. Inbound auth (`require_api_key`) MUST, when enabled, require a valid client
   key on `/v1/*` via `X-API-Key` or `Authorization: Bearer`, returning 401
   otherwise; health/metrics/docs MUST remain open. It MUST fail closed.
6. Inbound rate limiting MUST apply a per-client sliding window to `/v1/*` and
   return 429 with `Retry-After` when exceeded.
7. CORS MUST NOT combine a wildcard origin with credentials.

## Non-Goals
This spec does not define the HTTP schema (`GATEWAY_API.md`) or discovery
probing (`MODEL_DISCOVERY.md`). It does not replace org security policy.

## Acceptance Evidence
- `tests/test_features.py` covers key-pool rotation/budget/cooldown/failover,
  masking in snapshots, inbound API-key auth (401/200), and rate-limit 429s.
- History contains no real key; `.env` is untracked (verified in PRs).

## Token Budget Class
Project contract. Load for work touching keys, auth, or inbound access.

## Related Specs
- `GLOBAL_SECURITY.md`
- `SECURITY_AND_SECRETS.md`
- `GATEWAY_API.md`

## AI Agent Directives
Never print, persist, or commit a key. Preserve masking and fail-closed auth.
Scan staged diffs for secrets before committing.
