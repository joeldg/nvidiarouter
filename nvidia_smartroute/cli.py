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


# @spec[PROJECT_PROFILE.md#Intent]
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


# @spec[PROJECT_PROFILE.md#Requirements]
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


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
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


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
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


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
def _gateway_healthy(health_url: str, timeout: float = 2.0) -> bool:
    """Return True if the gateway responds 200 at its health endpoint."""
    import httpx

    try:
        return httpx.get(health_url, timeout=timeout).status_code == 200
    except Exception:
        return False


# @spec[PROJECT_PROFILE.md#Requirements]
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


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
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


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
# @spec[PROJECT_PROFILE.md#Token Budget Class]
@app.command()
def config():
    """Show the current configuration."""
    console.print("[blue]Current Configuration[/blue]")
    config_dict = settings.model_dump()
    for key, value in config_dict.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            value = "[REDACTED]"
        console.print(f"{key}: {value}")


# @spec[PROJECT_PROFILE.md#Requirements]
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


# @spec[PROJECT_PROFILE.md#Requirements]
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


# @spec[PROJECT_PROFILE.md#Token Budget Class]
@app.command()
def version():
    """Show the version of NVIDIA SmartRoute."""
    console.print(f"NVIDIA SmartRoute CLI version: {__version__}")


if __name__ == "__main__":
    app()
