# @spec[PARKOUR_RESEARCH.md#Requirements]
"""Bounded, server-owned web-research lane for PARKOUR workers.

This module is the single egress path for PARKOUR research. It is disabled by
default and, when enabled, exposes only the narrow ``parkour_web_search``
capability: a query string, optional domain filters, and a bounded result set of
``(url, title, snippet)`` citations. It never returns raw page bodies, request
or response headers, cookies, or redirect chains to workers, never exposes the
provider key or ``Authorization`` header, and blocks private, loopback,
link-local, and otherwise non-public network targets (SSRF protection) before
any network call — including after DNS resolution and across redirects.
"""

import ipaddress
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

# Only these schemes/ports are reachable by the research lane. HTTPS-only keeps
# egress on the standard TLS port and avoids opportunistic plaintext.
_ALLOWED_SCHEMES = frozenset({"https"})
_ALLOWED_PORTS = frozenset({None, 443})

# Resolver signature matches ``socket.getaddrinfo`` so tests can inject a fake.
Resolver = Callable[..., Sequence[Tuple[Any, ...]]]


class ResearchError(RuntimeError):
    """Base error for the research lane."""


class ResearchLimitError(ResearchError):
    """Raised when a research resource limit is reached."""


class ResearchBlockedError(ResearchError):
    """Raised when a target is disallowed by SSRF or domain policy."""


# @spec[PARKOUR_RESEARCH.md#Requirements]
def mask_secret(text: str, secret: Optional[str]) -> str:
    """Replace any occurrence of ``secret`` in ``text`` with a fixed mask."""
    if not text or not secret:
        return text
    return text.replace(secret, "***")


# @spec[PARKOUR_RESEARCH.md#Requirements]
def domain_allowed(host: str, allow: Sequence[str], block: Sequence[str]) -> bool:
    """Apply the allow/block suffix policy to a host. Block always wins."""
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return False

    def _matches(domain: str) -> bool:
        domain = domain.strip().lower().lstrip(".")
        return bool(domain) and (host == domain or host.endswith("." + domain))

    if any(_matches(d) for d in block):
        return False
    if allow and not any(_matches(d) for d in allow):
        return False
    return True


def _address_is_public(raw_ip: str) -> bool:
    """Return whether a resolved IP is a globally routable public address."""
    try:
        ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        return False
    # is_global is the positive test; the explicit checks below reject ranges
    # (e.g. some reserved/ULA space) that older Python is_global misses.
    if any((
        ip.is_private, ip.is_loopback, ip.is_link_local, ip.is_multicast,
        ip.is_reserved, ip.is_unspecified,
    )):
        return False
    return ip.is_global


# @spec[PARKOUR_RESEARCH.md#Requirements]
def guard_url(
    url: str,
    allow: Sequence[str] = (),
    block: Sequence[str] = (),
    resolver: Resolver = socket.getaddrinfo,
) -> str:
    """Validate one URL against scheme, credential, port, domain, and SSRF rules.

    Returns the validated host on success; raises ``ResearchBlockedError``
    before any network use otherwise. Resolution is performed here so that DNS
    answers pointing at private/internal addresses (rebinding) are rejected.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ResearchBlockedError(f"scheme not allowed: {parts.scheme or '(none)'}")
    if parts.username or parts.password:
        raise ResearchBlockedError("credentials in URL are not allowed")
    host = parts.hostname
    if not host:
        raise ResearchBlockedError("missing host")
    try:
        port = parts.port
    except ValueError as exc:
        raise ResearchBlockedError("invalid port") from exc
    if port not in _ALLOWED_PORTS:
        raise ResearchBlockedError(f"port not allowed: {port}")
    if not domain_allowed(host, allow, block):
        raise ResearchBlockedError(f"domain not allowed: {host}")
    _assert_public_host(host, port or 443, resolver)
    return host


def _assert_public_host(host: str, port: int, resolver: Resolver) -> None:
    """Resolve a host and require every returned address to be public."""
    try:
        ipaddress.ip_address(host)
        literals = [host]
    except ValueError:
        try:
            infos = resolver(host, port, proto=socket.IPPROTO_TCP)
        except OSError as exc:
            raise ResearchBlockedError(f"cannot resolve host: {host}") from exc
        literals = [info[4][0] for info in infos if info and info[4]]
        if not literals:
            raise ResearchBlockedError(f"host did not resolve: {host}")
    for raw_ip in literals:
        if not _address_is_public(raw_ip):
            raise ResearchBlockedError(f"non-public address for {host}: {raw_ip}")


# @spec[PARKOUR_RESEARCH.md#Requirements]
class Citation(BaseModel):
    """One bounded, sourced research result returned to a worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    title: str = ""
    snippet: str = ""


# @spec[PARKOUR_RESEARCH.md#Requirements]
class ResearchResult(BaseModel):
    """The bounded outcome of one ``parkour_web_search`` call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    citations: Tuple[Citation, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class RawResult:
    """A provider-returned candidate, before bounding/guarding by the session."""

    url: str
    title: str = ""
    snippet: str = ""


class SearchProvider(Protocol):
    """A pluggable search backend. The default talks to a configured HTTPS API.

    Implementations MUST NOT return the provider key or authorization headers in
    their results; the session additionally masks and bounds everything.
    """

    async def search(
        self, query: str, max_results: int, timeout: float, domains: Sequence[str]
    ) -> Sequence[RawResult]:
        ...


# @spec[PARKOUR_RESEARCH.md#Requirements]
@dataclass(frozen=True)
class ResearchLimits:
    """Hard per-run/per-node research bounds, enforced during execution."""

    max_searches_per_run: int
    max_searches_per_node: int
    max_query_chars: int
    max_results: int
    snippet_chars: int
    max_bytes: int
    timeout_seconds: float
    cost_per_search_usd: float
    max_cost_usd: float

    @classmethod
    def from_settings(cls, settings: Any) -> "ResearchLimits":
        return cls(
            max_searches_per_run=settings.parkour_research_max_searches_per_run,
            max_searches_per_node=settings.parkour_research_max_searches_per_node,
            max_query_chars=settings.parkour_research_max_query_chars,
            max_results=settings.parkour_research_max_results,
            snippet_chars=settings.parkour_research_snippet_chars,
            max_bytes=settings.parkour_research_max_bytes,
            timeout_seconds=settings.parkour_research_timeout_seconds,
            cost_per_search_usd=settings.parkour_research_cost_per_search_usd,
            max_cost_usd=settings.parkour_research_max_cost_usd,
        )


ProgressCallback = Callable[[Dict[str, Any]], Any]


# @spec[PARKOUR_RESEARCH.md#Requirements]
class ResearchSession:
    """Per-run research state: limits, dedup cache, and telemetry accounting.

    A session is bounded to a single PARKOUR run. Identical queries are
    deduplicated within the run so parallel workers do not stampede the
    provider. Nothing is persisted across runs or processes.
    """

    def __init__(
        self,
        limits: ResearchLimits,
        provider: SearchProvider,
        allow: Sequence[str] = (),
        block: Sequence[str] = (),
        telemetry: Optional["ResearchTelemetry"] = None,
        api_key: Optional[str] = None,
        resolver: Resolver = socket.getaddrinfo,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limits = limits
        self.provider = provider
        self.allow = tuple(allow)
        self.block = tuple(block)
        self.telemetry = telemetry
        self._api_key = api_key
        self._resolver = resolver
        self._clock = clock
        self._started = clock()
        self._searches = 0
        self._per_node: Dict[str, int] = {}
        self._bytes = 0
        self.cost_usd = 0.0
        self._cache: Dict[Tuple[str, Tuple[str, ...]], ResearchResult] = {}
        self._domains_seen: set = set()

    @property
    def searches(self) -> int:
        return self._searches

    @property
    def bytes_retained(self) -> int:
        return self._bytes

    def _mask(self, text: str) -> str:
        return mask_secret(text, self._api_key)

    async def search(
        self,
        query: str,
        node_id: str = "",
        domains: Optional[Sequence[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> ResearchResult:
        """Run one bounded ``parkour_web_search`` under all run/node limits."""
        query = self._validate_query(query)
        domain_filters = tuple((domains or ()))
        cache_key = (query, domain_filters)
        if cache_key in self._cache:
            if self.telemetry:
                self.telemetry.record_cache_hit()
            return self._cache[cache_key]

        self._enforce_pre_call(node_id)
        await self._emit(progress, {
            "type": "research_query_started",
            "node_id": node_id,
            "query_chars": len(query),
        })
        self._searches += 1
        self._per_node[node_id] = self._per_node.get(node_id, 0) + 1
        self.cost_usd += self.limits.cost_per_search_usd
        if self.cost_usd > self.limits.max_cost_usd:
            if self.telemetry:
                self.telemetry.record_limit_stop()
            raise ResearchLimitError("research cost limit exceeded")

        started = self._clock()
        raw = await self._call_provider(query, domain_filters, node_id, progress)
        result = self._shape(query, raw)
        self._cache[cache_key] = result
        if self.telemetry:
            self.telemetry.record_search(
                latency_ms=(self._clock() - started) * 1000,
                results=len(result.citations),
                bytes_retained=sum(len(c.snippet) for c in result.citations),
                domains={_host_of(c.url) for c in result.citations},
                truncated=result.truncated,
                cost_usd=self.limits.cost_per_search_usd,
            )
        await self._emit(progress, {
            "type": "research_query_completed",
            "node_id": node_id,
            "results": len(result.citations),
            "truncated": result.truncated,
        })
        return result

    def _validate_query(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            raise ResearchLimitError("research query is empty")
        if len(query) > self.limits.max_query_chars:
            raise ResearchLimitError(
                f"research query has {len(query)} chars; "
                f"limit is {self.limits.max_query_chars}"
            )
        return query

    async def _call_provider(
        self,
        query: str,
        domain_filters: Tuple[str, ...],
        node_id: str,
        progress: Optional[ProgressCallback],
    ) -> Sequence[RawResult]:
        """Call the provider, converting/redacting any failure without leaks."""
        try:
            return await self.provider.search(
                query, self.limits.max_results,
                self.limits.timeout_seconds, domain_filters,
            )
        except ResearchError:
            if self.telemetry:
                self.telemetry.record_failure()
            raise
        except Exception as exc:  # provider/transport error — never leak secrets
            if self.telemetry:
                self.telemetry.record_failure()
            await self._emit(progress, {
                "type": "research_query_failed",
                "node_id": node_id,
                "error": self._mask(str(exc) or repr(exc))[:200],
            })
            raise ResearchError("research provider call failed") from exc

    def _enforce_pre_call(self, node_id: str) -> None:
        if self._clock() - self._started > self.limits.timeout_seconds:
            if self.telemetry:
                self.telemetry.record_limit_stop()
            raise ResearchLimitError("research wall-clock budget exceeded")
        if self._searches >= self.limits.max_searches_per_run:
            if self.telemetry:
                self.telemetry.record_limit_stop()
            raise ResearchLimitError("research per-run search limit exceeded")
        if self._per_node.get(node_id, 0) >= self.limits.max_searches_per_node:
            if self.telemetry:
                self.telemetry.record_limit_stop()
            raise ResearchLimitError("research per-node search limit exceeded")

    def _shape(self, query: str, raw: Sequence[RawResult]) -> ResearchResult:
        """Guard, truncate, and byte-bound provider candidates into citations."""
        citations: List[Citation] = []
        truncated = False
        for item in raw:
            if len(citations) >= self.limits.max_results:
                truncated = True
                break
            try:
                guard_url(item.url, self.allow, self.block, self._resolver)
            except ResearchBlockedError:
                truncated = True
                continue  # drop non-public / disallowed results silently
            snippet = self._mask(item.snippet or "")[: self.limits.snippet_chars]
            if len(snippet) < len(item.snippet or ""):
                truncated = True
            if self._bytes + len(snippet) > self.limits.max_bytes:
                snippet = snippet[: max(0, self.limits.max_bytes - self._bytes)]
                truncated = True
            self._bytes += len(snippet)
            citations.append(Citation(
                url=item.url,
                title=self._mask(item.title or "")[: self.limits.snippet_chars],
                snippet=snippet,
            ))
            self._domains_seen.add(_host_of(item.url))
            if self._bytes >= self.limits.max_bytes:
                truncated = True
                break
        return ResearchResult(query=query, citations=tuple(citations), truncated=truncated)

    async def _emit(
        self, progress: Optional[ProgressCallback], event: Dict[str, Any]
    ) -> None:
        if progress is None:
            return
        outcome = progress(event)
        if hasattr(outcome, "__await__"):
            await outcome


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


# @spec[PARKOUR_RESEARCH.md#Requirements]
class ResearchTelemetry:
    """Bounded in-process telemetry for the research lane."""

    def __init__(self) -> None:
        self.searches = 0
        self.failures = 0
        self.cache_hits = 0
        self.limit_stops = 0
        self.results_retained = 0
        self.bytes_retained = 0
        self.truncations = 0
        self.total_latency_ms = 0.0
        self.total_cost_usd = 0.0
        self.domains: set = set()

    def record_search(
        self, latency_ms: float, results: int, bytes_retained: int,
        domains: set, truncated: bool, cost_usd: float,
    ) -> None:
        self.searches += 1
        self.results_retained += results
        self.bytes_retained += bytes_retained
        self.truncations += int(truncated)
        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost_usd
        self.domains.update(d for d in domains if d)

    def record_failure(self) -> None:
        self.failures += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_limit_stop(self) -> None:
        self.limit_stops += 1

    def snapshot(self) -> Dict[str, Any]:
        avg_latency = (
            self.total_latency_ms / self.searches if self.searches else 0.0
        )
        return {
            "searches": self.searches,
            "failures": self.failures,
            "cache_hits": self.cache_hits,
            "limit_stops": self.limit_stops,
            "results_retained": self.results_retained,
            "bytes_retained": self.bytes_retained,
            "truncations": self.truncations,
            "distinct_domains": len(self.domains),
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(self.total_cost_usd, 8),
        }


research_telemetry = ResearchTelemetry()


# @spec[PARKOUR_RESEARCH.md#Requirements]
class HttpSearchProvider:
    """Generic HTTPS search provider guarded by the SSRF egress rules.

    Posts ``{"query": ..., "max_results": ...}`` to a configured endpoint over
    HTTPS with the provider key in the ``Authorization`` header, following no
    redirects. The endpoint host is guarded before the call, and the provider
    key/header are never returned to workers. Response parsing accepts the
    common ``{"results": [{"url"|"link", "title", "snippet"|"content"}, ...]}``
    shape.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str],
        allow: Sequence[str] = (),
        block: Sequence[str] = (),
        resolver: Resolver = socket.getaddrinfo,
    ):
        self.endpoint = endpoint
        self._api_key = api_key
        self.allow = tuple(allow)
        self.block = tuple(block)
        self._resolver = resolver

    async def search(
        self, query: str, max_results: int, timeout: float, domains: Sequence[str]
    ) -> Sequence[RawResult]:
        import httpx

        # Guard the provider endpoint itself (allow-listing does not apply to the
        # provider host; only SSRF/scheme/port rules do).
        guard_url(self.endpoint, (), self.block, self._resolver)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: Dict[str, Any] = {"query": query, "max_results": max_results}
        if domains:
            payload["include_domains"] = list(domains)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False
        ) as client:
            resp = await client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        raw: List[RawResult] = []
        for item in (data.get("results") or []) if isinstance(data, dict) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or item.get("link") or "").strip()
            if not url:
                continue
            raw.append(RawResult(
                url=url,
                title=str(item.get("title") or ""),
                snippet=str(item.get("snippet") or item.get("content") or ""),
            ))
        return raw


# @spec[PARKOUR_RESEARCH.md#Requirements]
def build_research_session(
    settings: Any,
    telemetry: Optional[ResearchTelemetry] = None,
    provider: Optional[SearchProvider] = None,
) -> Optional[ResearchSession]:
    """Build a per-run session, or ``None`` when research is unavailable.

    Returns ``None`` when the research lane is disabled or no provider is
    configured, so callers can leave the ordinary PARKOUR path untouched.
    """
    if not getattr(settings, "enable_parkour_research", False):
        return None
    allow = settings.parkour_research_allowlist
    block = settings.parkour_research_blocklist
    api_key = settings.parkour_research_api_key
    if provider is None:
        endpoint = settings.parkour_research_endpoint
        if not endpoint:
            return None
        provider = HttpSearchProvider(endpoint, api_key, allow, block)
    return ResearchSession(
        ResearchLimits.from_settings(settings),
        provider,
        allow=allow,
        block=block,
        telemetry=telemetry if telemetry is not None else research_telemetry,
        api_key=api_key,
    )
