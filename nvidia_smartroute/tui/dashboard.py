# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
"""
Rich Terminal User Interface (TUI) dashboard for NVIDIA-SmartRoute-CLI.

An interactive OpenShell-style console that polls the running gateway's
``/metrics`` endpoint and surfaces real-time state:

  * active local connections + a requests/sec sparkline
  * per-model throughput, latency, and cost (performance table)
  * live routing decision log

Launch with ``nvidia-smartroute dashboard`` while the gateway is running.
"""

import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Sparkline, Static

from ..config import settings

_HISTORY = 60  # samples in the requests/sec sparkline


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
class DashboardApp(App):
    """Textual dashboard polling the gateway's live metrics."""

    CSS = """
    Screen { layout: vertical; background: $surface; }
    #summary { height: 3; padding: 1 2; background: $panel; color: $text; }
    #chart { height: 6; border: round $accent; padding: 0 1; margin: 0 1; }
    #tables { height: 1fr; }
    #models { width: 2fr; border: round $primary; margin: 0 1; }
    #log { width: 1fr; border: round $secondary; margin: 0 1; }
    .title { text-style: bold; color: $accent; }
    Sparkline { height: 3; margin-top: 1; }
    Sparkline > .sparkline--max-color { color: $success; }
    Sparkline > .sparkline--min-color { color: $primary; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_now", "Refresh"),
    ]

    def __init__(
        self,
        metrics_url: Optional[str] = None,
        refresh_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        host = "127.0.0.1" if settings.host in ("0.0.0.0", "") else settings.host
        self.metrics_url = metrics_url or f"http://{host}:{settings.port}/metrics"
        self.refresh_rate = refresh_rate or settings.tui_refresh_rate
        self._client = httpx.AsyncClient(timeout=5.0)
        self._seen_log_ids: set[str] = set()
        self._req_history: deque = deque([0.0] * _HISTORY, maxlen=_HISTORY)
        self._last_total: Optional[int] = None
        self._last_t: Optional[float] = None

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Connecting to gateway...", id="summary")
        with Vertical(id="chart"):
            yield Static("Requests/sec", classes="title", id="chart_title")
            yield Sparkline(list(self._req_history), summary_function=max, id="req_spark")
        with Horizontal(id="tables"):
            with Vertical(id="models"):
                yield Static("Model Performance", classes="title")
                yield DataTable(id="model_table")
            with Vertical(id="log"):
                yield Static("Routing Log", classes="title")
                yield RichLog(id="routing_log", highlight=True, markup=True)
        yield Footer()

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    def on_mount(self) -> None:
        self.title = "NVIDIA-SmartRoute-CLI"
        self.sub_title = f"gateway @ {self.metrics_url}"
        table = self.query_one("#model_table", DataTable)
        table.add_columns(
            "Model", "Params", "Reqs", "Avg ms", "Tok/s", "Max t/s", "Cost $", "Errors"
        )
        table.zebra_stripes = True
        self.set_interval(self.refresh_rate, self.refresh_metrics)
        self.call_after_refresh(self.refresh_metrics)

    # @spec[PROJECT_PROFILE.md#Acceptance Evidence]
    async def refresh_metrics(self) -> None:
        """Poll the gateway and update the widgets."""
        try:
            response = await self._client.get(self.metrics_url)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # gateway down or unreachable
            self.query_one("#summary", Static).update(
                f"[red]Unable to reach gateway at {self.metrics_url}: {exc}[/red]"
            )
            return
        self._update_rate(data)
        self._update_summary(data)
        self._update_models(data)
        self._update_log(data)

    def _update_rate(self, data: Dict[str, Any]) -> None:
        """Compute requests/sec from total-request deltas and feed the chart."""
        total = int(data.get("total_requests", 0))
        now = time.time()
        if self._last_total is not None and self._last_t is not None:
            dt = max(now - self._last_t, 1e-6)
            rate = max(0.0, (total - self._last_total) / dt)
            self._req_history.append(round(rate, 2))
            self.query_one("#req_spark", Sparkline).data = list(self._req_history)
            peak = max(self._req_history)
            self.query_one("#chart_title", Static).update(
                f"Requests/sec  (now {rate:.1f}  peak {peak:.1f})"
            )
        self._last_total = total
        self._last_t = now

    def _update_summary(self, data: Dict[str, Any]) -> None:
        uptime = int(data.get("uptime_seconds", 0))
        cache = data.get("cache", {}) or {}
        cost = data.get("total_cost_usd", 0.0)
        conc = data.get("concurrency", {}) or {}
        summary = (
            f"[b]Conns:[/b] {data.get('active_connections', 0)}   "
            f"[b]Requests:[/b] {data.get('total_requests', 0)}   "
            f"[b]In-flight:[/b] {conc.get('inflight', 0)}   "
            f"[b]Cache:[/b] {cache.get('hit_rate', 0) * 100:.0f}%   "
            f"[b]Cost:[/b] ${cost:.4f}   "
            f"[b]Uptime:[/b] {uptime}s   [b]Port:[/b] {settings.port}"
        )
        self.query_one("#summary", Static).update(summary)

    def _update_models(self, data: Dict[str, Any]) -> None:
        table = self.query_one("#model_table", DataTable)
        table.clear()
        for m in data.get("models", []):
            params = m.get("parameters_b", 0) or 0
            table.add_row(
                m.get("model_id", "?"),
                f"{params:.0f}B" if params else "?",
                str(m.get("request_count", 0)),
                f"{m.get('avg_latency_ms', 0):.0f}",
                f"{m.get('throughput_tps', 0):.1f}",
                f"{m.get('max_tps', 0):.1f}",
                f"{m.get('total_cost_usd', 0):.4f}",
                str(m.get("error_count", 0)),
            )

    def _update_log(self, data: Dict[str, Any]) -> None:
        log = self.query_one("#routing_log", RichLog)
        for entry in data.get("routing_log", []):
            entry_id = entry.get("request_id", "")
            if entry_id in self._seen_log_ids:
                continue
            self._seen_log_ids.add(entry_id)
            ts = datetime.fromtimestamp(entry.get("timestamp", time.time())).strftime("%H:%M:%S")
            log.write(
                f"[dim]{ts}[/dim] [cyan]{entry.get('task_type')}[/cyan] -> "
                f"[green]{entry.get('model')}[/green] "
                f"(conf {entry.get('confidence')})"
            )

    def action_refresh_now(self) -> None:
        self.run_worker(self.refresh_metrics())

    async def on_unmount(self) -> None:
        await self._client.aclose()


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
def run_dashboard(metrics_url: Optional[str] = None, refresh_rate: Optional[float] = None) -> None:
    """Entry point used by the CLI to launch the dashboard."""
    DashboardApp(metrics_url=metrics_url, refresh_rate=refresh_rate).run()
