# @spec[PROJECT_PROFILE.md]
"""
Command-line interface for NVIDIA-SmartRoute-CLI.
"""

import signal
import sys
import uvicorn

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from nvidia_smartroute import __version__
from nvidia_smartroute.config import settings

app = typer.Typer(help="NVIDIA SmartRoute CLI - AI-powered API gateway")
console = Console()


# @spec[GATEWAY_API.md#Requirements]
def _display_banner():
    """Display the application banner."""
    banner = r"""
  _____                      __  ____              __
  / ___/____ ___  ____ ______/ /_/ __ \____  __  __/ /____
  \__ \/ __ `__ \/ __ `/ ___/ __/ /_/ / __ \/ / / / __/ _ \
 ___/ / / / / / / /_/ / /  / /_/ _, _/ /_/ / /_/ / /_/  __/
/____/_/ /_/ /_/\__,_/_/   \__/_/ |_|\____/\__,_/\__/\___/
    """
    styled_banner = Text(banner, style="cyan")
    console.print(Panel(styled_banner, title="NVIDIA SmartRoute", border_style="blue"))


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def start(
    host: str = typer.Option(settings.host, help="Host to bind to"),
    port: int = typer.Option(settings.port, help="Port to bind to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the NVIDIA SmartRoute gateway server."""
    import atexit
    import os
    from pathlib import Path

    _display_banner()
    console.print(f"[green]Starting NVIDIA SmartRoute gateway on {host}:{port}[/green]")
    console.print(f"[blue]Workers:[/blue] {workers}")
    console.print(f"[blue]Reload:[/blue] {reload}")

    # Write a PID file so `stop` can signal this process; clean it up on exit.
    pid_path = Path(settings.pid_file)
    pid_path.write_text(str(os.getpid()))
    console.print(f"[blue]PID file:[/blue] {pid_path} ({os.getpid()})")

    def _cleanup_pidfile():
        try:
            if pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass

    atexit.register(_cleanup_pidfile)

    # Configure signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        console.print("\n[yellow]Received shutdown signal. Stopping gracefully...[/yellow]")
        _cleanup_pidfile()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(
        "nvidia_smartroute.gateway.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def stop():
    """Stop the running NVIDIA SmartRoute gateway (via its PID file)."""
    import os
    import signal as _signal
    from pathlib import Path

    pid_path = Path(settings.pid_file)
    if not pid_path.exists():
        console.print(f"[yellow]No PID file at {pid_path}; gateway not running?[/yellow]")
        raise typer.Exit(code=1)

    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError) as exc:
        console.print(f"[red]Could not read PID file: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[yellow]Stopping NVIDIA SmartRoute gateway (pid {pid})...[/yellow]")
    try:
        os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        console.print("[yellow]Process not found; removing stale PID file.[/yellow]")
        pid_path.unlink(missing_ok=True)
        raise typer.Exit(code=1)
    except PermissionError:
        console.print("[red]Permission denied signaling the process.[/red]")
        raise typer.Exit(code=1)

    console.print("[green]Sent SIGTERM. Gateway is shutting down.[/green]")


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def status(
    host: str = typer.Option(settings.host, help="Gateway host to probe"),
    port: int = typer.Option(settings.port, help="Gateway port to probe"),
):
    """Show the current status of the NVIDIA SmartRoute gateway."""
    import httpx

    console.print("[blue]NVIDIA SmartRoute Gateway Status[/blue]")
    console.print(f"Host: {settings.host}")
    console.print(f"Port: {settings.port}")
    console.print(f"Workers: {settings.workers}")
    console.print(f"Log Level: {settings.log_level}")

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{probe_host}:{port}/health"
    try:
        resp = httpx.get(url, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        console.print(f"[green]Status: RUNNING[/green] ({data.get('status')})")
    except Exception as exc:
        console.print(f"[red]Status: NOT RUNNING[/red] (could not reach {url})")
        console.print(f"[dim]{exc}[/dim]")


# @spec[GATEWAY_API.md#Requirements]
def _gateway_healthy(health_url: str, timeout: float = 2.0) -> bool:
    """Return True if the gateway responds 200 at its health endpoint."""
    import httpx

    try:
        return httpx.get(health_url, timeout=timeout).status_code == 200
    except Exception:
        return False


# @spec[GATEWAY_API.md#Requirements]
def _start_gateway(host: str, port: int, health_url: str, wait_seconds: int = 40):
    """
    Launch the gateway as a background subprocess and wait until it is ready.

    Returns the Popen handle on success. Exits the CLI on failure.
    """
    import subprocess
    import time as _time

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "nvidia_smartroute.gateway.server:app",
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = _time.time() + wait_seconds
    while _time.time() < deadline:
        if proc.poll() is not None:
            console.print("[red]Gateway process exited before becoming ready.[/red]")
            raise typer.Exit(code=1)
        if _gateway_healthy(health_url):
            console.print("[green]Gateway is ready.[/green]")
            return proc
        _time.sleep(0.5)

    console.print(f"[red]Gateway did not become ready within {wait_seconds}s.[/red]")
    proc.terminate()
    raise typer.Exit(code=1)


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def dashboard(
    host: str = typer.Option(settings.host, help="Gateway host to connect to"),
    port: int = typer.Option(settings.port, help="Gateway port to connect to"),
    refresh: float = typer.Option(
        settings.tui_refresh_rate, help="Refresh interval in seconds"
    ),
    start_gateway: bool = typer.Option(
        True,
        "--start-gateway/--no-start-gateway",
        help="Start the gateway automatically if it is not already running",
    ),
):
    """Launch the interactive real-time metrics dashboard (TUI).

    If the gateway is not already running it is started automatically (unless
    --no-start-gateway is passed) and shut down again when the dashboard exits.
    """
    from nvidia_smartroute.tui import run_dashboard

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    health_url = f"http://{probe_host}:{port}/health"
    metrics_url = f"http://{probe_host}:{port}/metrics"

    proc = None
    if _gateway_healthy(health_url):
        console.print(f"[green]Gateway already running at {health_url}[/green]")
    elif start_gateway:
        console.print(
            f"[yellow]Gateway not running; starting it on {host}:{port}...[/yellow]"
        )
        proc = _start_gateway(host, port, health_url)
    else:
        console.print(
            f"[yellow]Gateway not reachable at {health_url}; "
            f"launching dashboard anyway.[/yellow]"
        )

    try:
        console.print(f"[green]Launching dashboard against {metrics_url}[/green]")
        run_dashboard(metrics_url=metrics_url, refresh_rate=refresh)
    finally:
        # Only tear down a gateway that this command started.
        if proc is not None:
            console.print("[yellow]Shutting down gateway started by dashboard...[/yellow]")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()


# @spec[GATEWAY_API.md#Requirements]
# @spec[GATEWAY_API.md#Requirements]
@app.command()
def config():
    """Show the current configuration."""
    console.print("[blue]Current Configuration[/blue]")
    config_dict = settings.model_dump()
    for key, value in config_dict.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            value = "[REDACTED]"
        console.print(f"{key}: {value}")


# @spec[MODEL_DISCOVERY.md#Requirements]
def _probe_model(model_id: str, headers: dict) -> str:
    """Send a 1-token chat request to check a model is servable for this account."""
    import httpx

    try:
        resp = httpx.post(
            f"{settings.nvidia_nim_base_url}/chat/completions",
            headers=headers,
            json={"model": model_id, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1},
            timeout=30.0,
        )
        if resp.status_code == 200:
            return "[green]servable[/green]"
        if resp.status_code == 404:
            return "[red]not available (404)[/red]"
        if resp.status_code == 401:
            return "[red]unauthorized (401)[/red]"
        return f"[yellow]HTTP {resp.status_code}[/yellow]"
    except Exception as exc:
        return f"[red]error: {type(exc).__name__}[/red]"


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def doctor(
    probe_models: bool = typer.Option(
        True, "--probe-models/--no-probe-models",
        help="Send a tiny request to each registered model to confirm it's servable",
    ),
):
    """Diagnose configuration, upstream connectivity, and model availability."""
    import httpx
    from rich.table import Table

    from nvidia_smartroute.routing.router import router

    console.print("[blue bold]NVIDIA SmartRoute — Doctor[/blue bold]\n")

    # 1) Configuration
    keys = settings.api_keys
    console.print("[bold]Configuration[/bold]")
    console.print(f"  API keys configured : {len(keys)}")
    console.print(f"  Base URL            : {settings.nvidia_nim_base_url}")
    console.print(f"  Gateway bind        : {settings.host}:{settings.port}")
    console.print(f"  Per-key rate limit  : {settings.rate_limit_per_key}/min")
    if not keys:
        console.print("  [red]No API key configured — set NVIDIA_API_KEY in .env[/red]")
        raise typer.Exit(code=1)

    headers = {"Authorization": f"Bearer {keys[0]}", "Accept": "application/json"}

    # 2) Connectivity
    console.print("\n[bold]Connectivity[/bold]")
    try:
        resp = httpx.get(
            f"{settings.nvidia_nim_base_url}/models", headers=headers, timeout=15.0
        )
        resp.raise_for_status()
        catalog = {m["id"] for m in resp.json().get("data", [])}
        console.print(f"  [green]OK[/green] — reached NIM ({len(catalog)} models in catalog)")
    except Exception as exc:
        console.print(f"  [red]FAILED[/red] — {exc}")
        raise typer.Exit(code=1)

    # 3) Registered model availability
    console.print("\n[bold]Registered models[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("In catalog")
    table.add_column("Servable" if probe_models else "Status")
    unservable = 0
    for model_id in router.model_registry.models:
        in_catalog = "[green]yes[/green]" if model_id in catalog else "[yellow]no[/yellow]"
        if probe_models:
            status = _probe_model(model_id, headers)
            if "servable" not in status:
                unservable += 1
        else:
            status = "[dim]skipped[/dim]"
        table.add_row(model_id, in_catalog, status)
    console.print(table)

    if probe_models and unservable:
        console.print(
            f"\n[yellow]{unservable} registered model(s) are not servable for this "
            f"account — routing will fall back to the others.[/yellow]"
        )
        raise typer.Exit(code=1)
    console.print("\n[green]Doctor complete — all good.[/green]")


# A varied prompt mix so different tasks/models light up on the dashboard.
# Some prompts repeat across a run, which also exercises the response cache.
# @spec[OBSERVABILITY.md#Requirements]
_STRESS_PROMPTS = [
    "What is 17 * 23?",
    "Write a Python function to reverse a string",
    "Write a haiku about the ocean",
    "Translate 'good morning' to Spanish",
    "Summarize what a CPU does in one sentence",
    "Hello, how are you today?",
    "Explain why the sky is blue",
    "What is 2 + 2?",
    "Solve for x: 3x + 5 = 20",
    "Write a limerick about coffee",
]


# @spec[GATEWAY_API.md#Requirements]
def _percentile(values, pct: float) -> float:
    """Return the pct-th percentile (0..100) of a list of numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


# @spec[OBSERVABILITY.md#Requirements]
def _summarize_stress(results: list, elapsed: float) -> dict:
    """Aggregate stress-run results into a stats dict (pure; unit-tested)."""
    total = len(results)
    ok = [r for r in results if r["status"] == 200]
    latencies = [r["ms"] for r in ok]
    status_counts: dict = {}
    model_counts: dict = {}
    cache_hits = 0
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        if r.get("model"):
            model_counts[r["model"]] = model_counts.get(r["model"], 0) + 1
        if r.get("cache") == "HIT":
            cache_hits += 1
    return {
        "total": total,
        "ok": len(ok),
        "failed": total - len(ok),
        "rps": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "p50_ms": round(_percentile(latencies, 50), 1),
        "p90_ms": round(_percentile(latencies, 90), 1),
        "p99_ms": round(_percentile(latencies, 99), 1),
        "status_counts": status_counts,
        "model_counts": model_counts,
        "cache_hits": cache_hits,
    }


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def stress(
    host: str = typer.Option(settings.host, help="Gateway host"),
    port: int = typer.Option(settings.port, help="Gateway port"),
    requests: int = typer.Option(100, "--requests", "-n", help="Total requests to send"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent requests"),
    max_tokens: int = typer.Option(16, help="max_tokens per request (keep small)"),
    rps: float = typer.Option(0.0, help="Throttle to this requests/sec (0 = unthrottled)"),
):
    """Drive load at a running gateway so you can watch the dashboard live.

    Run `nvidia-smartroute dashboard` in one terminal, then this in another.
    """
    import asyncio
    import time

    import httpx
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{probe_host}:{port}/v1/chat/completions"

    if not _gateway_healthy(f"http://{probe_host}:{port}/health"):
        console.print(f"[red]Gateway not reachable at {probe_host}:{port}.[/red]")
        console.print("[yellow]Start it first: nvidia-smartroute dashboard[/yellow]")
        raise typer.Exit(code=1)

    console.print(
        f"[green]Stressing[/green] {url}\n"
        f"  requests={requests} concurrency={concurrency} "
        f"max_tokens={max_tokens} rps={'∞' if not rps else rps}\n"
    )

    results: list = []
    interval = 1.0 / rps if rps else 0.0

    async def _run(progress, task_id):
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=180.0) as client:
            async def one(i: int):
                prompt = _STRESS_PROMPTS[i % len(_STRESS_PROMPTS)]
                async with sem:
                    t0 = time.time()
                    try:
                        resp = await client.post(
                            url,
                            json={
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": max_tokens,
                                "temperature": 0,
                            },
                        )
                        results.append({
                            "status": resp.status_code,
                            "ms": (time.time() - t0) * 1000.0,
                            "model": resp.headers.get("X-Selected-Model"),
                            "task": resp.headers.get("X-Task-Type"),
                            "cache": resp.headers.get("X-Cache"),
                        })
                    except Exception as exc:
                        results.append({
                            "status": 0, "ms": (time.time() - t0) * 1000.0,
                            "model": None, "task": None, "cache": None,
                            "error": type(exc).__name__,
                        })
                    progress.advance(task_id)

            tasks = []
            for i in range(requests):
                tasks.append(asyncio.create_task(one(i)))
                if interval:
                    await asyncio.sleep(interval)
            await asyncio.gather(*tasks)

    start = time.time()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("firing", total=requests)
        asyncio.run(_run(progress, task_id))
    elapsed = time.time() - start

    stats = _summarize_stress(results, elapsed)

    console.print(
        f"\n[bold]Results[/bold] — {stats['ok']}/{stats['total']} ok "
        f"({stats['failed']} failed) in {elapsed:.1f}s @ "
        f"[cyan]{stats['rps']} req/s[/cyan]"
    )
    console.print(
        f"  latency (ok): p50 {stats['p50_ms']}ms | "
        f"p90 {stats['p90_ms']}ms | p99 {stats['p99_ms']}ms"
    )
    console.print(f"  status codes: {stats['status_counts']}  cache hits: {stats['cache_hits']}")

    if stats["model_counts"]:
        table = Table(title="Routed models", show_header=True, header_style="bold")
        table.add_column("Model")
        table.add_column("Requests", justify="right")
        for model, count in sorted(stats["model_counts"].items(), key=lambda kv: -kv[1]):
            table.add_row(model, str(count))
        console.print(table)

    console.print("\n[dim]Watch live detail on the dashboard's /metrics view.[/dim]")


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def discover(  # noqa: C901
    output: str = typer.Option(settings.models_file, help="Where to write discovered models"),
    probe: bool = typer.Option(
        True, "--probe/--no-probe",
        help="Probe each model for servability (accurate but slower)",
    ),
    limit: int = typer.Option(0, help="Only check the first N catalog models (0 = all)"),
    include_embeddings: bool = typer.Option(False, help="Include embedding models"),
    delay: float = typer.Option(
        -1.0,
        help="Seconds between probes (default: derived from the per-key rate limit)",
    ),
):
    """Discover which NIM models your account can serve and register them.

    Writes a capability profile per servable model; restart the gateway to route
    across the discovered set. Probing is throttled to stay under the per-key
    rate limit, so a full catalog scan takes a few minutes — use --limit to sample.
    """
    import time

    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table

    from nvidia_smartroute import discovery
    from nvidia_smartroute.model_catalog import is_routable, rank_by_capability

    # Throttle to stay under the per-key rate limit (with a little headroom).
    if delay < 0:
        delay = 60.0 / max(1, settings.rate_limit_per_key) + 0.2

    keys = settings.api_keys
    if not keys:
        console.print("[red]No API key configured — set NVIDIA_API_KEY in .env[/red]")
        raise typer.Exit(code=1)
    key = keys[0]
    base = settings.nvidia_nim_base_url

    console.print("[blue bold]Discovering NIM models[/blue bold]")
    try:
        catalog = discovery.fetch_catalog(base, key)
    except Exception as exc:
        console.print(f"[red]Failed to fetch catalog: {exc}[/red]")
        raise typer.Exit(code=1)
    if not include_embeddings:
        catalog = [m for m in catalog if is_routable(m)]
    if limit:
        catalog = catalog[:limit]
    est = f"  ~{len(catalog) * delay / 60:.1f} min at {delay:.1f}s/probe" if probe else ""
    console.print(f"  {len(catalog)} models to check "
                  f"({'probing' if probe else 'no probe'}){est}\n")

    caps = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("checking", total=len(catalog))
        for i, model_id in enumerate(catalog):
            ok = discovery.probe_servable(base, key, model_id) if probe else True
            if ok:
                caps.append(discovery.deserialize(discovery.infer_capability(model_id)))
            progress.advance(task_id)
            # Throttle between probes (not after the last one).
            if probe and delay and i < len(catalog) - 1:
                time.sleep(delay)

    if not caps:
        console.print("[yellow]No servable models found.[/yellow]")
        raise typer.Exit(code=1)

    discovery.save_models(output, caps)

    profiles = rank_by_capability([discovery.serialize(c) for c in caps])
    table = Table(title=f"{len(caps)} servable models", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Params (B)", justify="right")
    table.add_column("Tasks")
    for p in profiles:
        params = f"{p['parameters_b']:.0f}" if p["parameters_b"] else "?"
        tasks = ", ".join(p["supported_tasks"][:3])
        if len(p["supported_tasks"]) > 3:
            tasks += ", …"
        table.add_row(p["model_id"], params, tasks)
    console.print(table)
    console.print(
        f"\n[green]Saved {len(caps)} models to {output}. "
        f"Restart the gateway to route across them.[/green]"
    )


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def benchmark(  # noqa: C901
    per_model: int = typer.Option(3, help="Requests per model"),
    max_tokens: int = typer.Option(48, help="max_tokens per request"),
    top: int = typer.Option(8, help="Benchmark the N largest models (0 = all)"),
    delay: float = typer.Option(
        -1.0, help="Seconds between requests (default: from the per-key rate limit)"
    ),
    save: bool = typer.Option(
        False, "--save", help="Write measured latency/throughput back to the model file"
    ),
):
    """Benchmark registered models directly against NIM: latency + throughput.

    Standalone (no gateway needed) — reads the router registry (built-in defaults
    plus anything from `discover`), calls each model directly, and ranks them so
    you can see which are fastest and most capable. Throttled to respect the
    per-key rate limit.
    """
    import time

    import httpx
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table

    from nvidia_smartroute.routing.router import router

    keys = settings.api_keys
    if not keys:
        console.print("[red]No API key configured — set NVIDIA_API_KEY in .env[/red]")
        raise typer.Exit(code=1)
    key = keys[0]
    url = f"{settings.nvidia_nim_base_url.rstrip('/')}/chat/completions"
    if delay < 0:
        delay = 60.0 / max(1, settings.rate_limit_per_key) + 0.2

    models = sorted(
        router.model_registry.models.values(),
        key=lambda m: m.parameters_b, reverse=True,
    )
    if top:
        models = models[:top]

    total_reqs = len(models) * per_model
    console.print(
        f"[green]Benchmarking[/green] {len(models)} models × {per_model} "
        f"(~{total_reqs * delay / 60:.1f} min at {delay:.1f}s/req)\n"
    )
    prompt = "Explain what an API gateway does in two short sentences."
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

    rows = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress, httpx.Client(timeout=180.0) as client:
        task_id = progress.add_task("benchmarking", total=total_reqs)
        for model in models:
            lats, ok, ctoks = [], 0, 0
            for i in range(per_model):
                t0 = time.time()
                try:
                    r = client.post(url, headers=headers, json={
                        "model": model.model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens, "temperature": 0,
                    })
                    if r.status_code == 200:
                        ok += 1
                        lats.append((time.time() - t0) * 1000.0)
                        u = r.json().get("usage", {}) or {}
                        ctoks += u.get("completion_tokens") or u.get("total_tokens") or 0
                except Exception:
                    pass
                progress.advance(task_id)
                if i < per_model - 1 or model is not models[-1]:
                    time.sleep(delay)
            total_s = sum(lats) / 1000.0
            rows.append({
                "model": model.model_id,
                "params": model.parameters_b,
                "ok": ok,
                "p50_ms": _percentile(lats, 50),
                "tps": round(ctoks / total_s, 1) if total_s > 0 else 0.0,
            })

    # Rank by success, then generation speed (tok/s).
    rows.sort(key=lambda r: (-r["ok"], -r["tps"]))

    table = Table(title="Model leaderboard (fastest first)", show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Params (B)", justify="right")
    table.add_column("OK", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("tok/s", justify="right")
    for r in rows:
        params = f"{r['params']:.0f}" if r["params"] else "?"
        table.add_row(
            r["model"], params, f"{r['ok']}/{per_model}",
            f"{r['p50_ms']:.0f}" if r["p50_ms"] else "-",
            f"{r['tps']}" if r["tps"] else "-",
        )
    console.print(table)
    winners = [r for r in rows if r["ok"] == per_model and r["tps"]]
    if winners:
        best = max(winners, key=lambda r: r["tps"])
        console.print(
            f"\n[green]Fastest reliable model:[/green] {best['model']} "
            f"({best['params']:.0f}B, {best['tps']} tok/s)"
        )

    # Feed measured performance back into routing.
    if save:
        import os

        from nvidia_smartroute import discovery

        if not os.path.exists(settings.models_file):
            console.print(
                f"\n[yellow]--save skipped: {settings.models_file} not found "
                f"(run `discover` first).[/yellow]"
            )
        else:
            caps = discovery.load_models(settings.models_file)
            measured = {
                r["model"]: {"ok": r["ok"] > 0, "p50_ms": r["p50_ms"], "tps": r["tps"]}
                for r in rows
            }
            n = discovery.apply_benchmark(caps, measured)
            discovery.save_models(settings.models_file, caps)
            console.print(
                f"\n[green]Saved measured latency/throughput for {n} model(s) to "
                f"{settings.models_file}.[/green] Restart the gateway to use it."
            )


# @spec[RECOMMENDATION.md#Requirements]
@app.command()
def recommend(
    task: str = typer.Option(None, help="Only recommend for this task type"),
):
    """Recommend the best model per task from the registry + live metrics.

    Read-only and standalone — no gateway required.
    """
    from rich.table import Table

    from nvidia_smartroute.recommend import is_task, recommend_all

    if task is not None and not is_task(task):
        console.print(f"[red]Unknown task '{task}'.[/red]")
        raise typer.Exit(code=1)

    recs = recommend_all()
    if task is not None:
        recs = {task: recs[task]}

    table = Table(title="Recommended model per task", show_header=True, header_style="bold")
    table.add_column("Task")
    table.add_column("Recommended model")
    table.add_column("Params", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Basis")
    table.add_column("Confidence", justify="right")
    table.add_column("$/1k", justify="right")
    for task_name, rec in recs.items():
        if not rec["model"]:
            table.add_row(task_name, "[dim]none[/dim]", "-", "-", "-", "-", "-")
            continue
        r = rec["rationale"]
        params = f"{r['parameters_b']:.0f}B" if r["parameters_b"] else "?"
        # Visibly mark low-confidence recommendations (RECOMMENDATION.md req.9).
        if rec["low_confidence"]:
            conf = f"[yellow]{rec['confidence']:.2f} ⚠ low[/yellow]"
        else:
            conf = f"{rec['confidence']:.2f}"
        table.add_row(
            task_name, rec["model"], params, f"{r['latency_ms']:.0f}ms",
            rec["basis"], conf, f"{r['output_cost_per_1k']}",
        )
    console.print(table)


# @spec[GATEWAY_API.md#Requirements]
@app.command()
def version():
    """Show the version of NVIDIA SmartRoute."""
    console.print(f"NVIDIA SmartRoute CLI version: {__version__}")


if __name__ == "__main__":
    app()
