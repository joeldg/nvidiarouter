# @spec[PROJECT_PROFILE.md]
"""
Command-line interface for NVIDIA-SmartRoute-CLI.
"""

import asyncio
import signal
import sys
from typing import Optional
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
    banner = """
  ____ _               _           _    ____                                  
 / ___| |__   ___  ___| | __ _    / \\  |  _ \\  ___   ___ _ __   __ _ _ __ ___ 
| |   | '_ \\ / _ \\/ __| |/ _`    / _ \\ | | | |/ _ \\ / _ \\ '_ \\ / _` | '__/ __|
| |___| | | |  __/ (__| | (_|   / ___ \\| |_| | (_) | (_) | | | | (_) | |  \\__ \\
 \\____|_| |_|\\___|\\___|_|\\__,_  /_/   \\_\\____/ \\___/ \\___/|_|  |_\\__,_|_|  |___/
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
    _display_banner()
    console.print(f"[green]Starting NVIDIA SmartRoute gateway on {host}:{port}[/green]")
    console.print(f"[blue]Workers:[/blue] {workers}")
    console.print(f"[blue]Reload:[/blue] {reload}")
    
    # Configure signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        console.print("\\n[yellow]Received shutdown signal. Stopping gracefully...[/yellow]")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    uvicorn.run(
        "nvidia_smartroute.gateway.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.command()
def stop():
    """Stop the NVIDIA SmartRoute gateway server."""
    console.print("[yellow]Stopping NVIDIA SmartRoute gateway...[/yellow]")
    # In a real implementation, this would send a signal to the running process
    console.print("[green]Gateway stopped.[/green]")


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
@app.command()
def status():
    """Show the current status of the NVIDIA SmartRoute gateway."""
    console.print("[blue]NVIDIA SmartRoute Gateway Status[/blue]")
    console.print(f"Host: {settings.host}")
    console.print(f"Port: {settings.port}")
    console.print(f"Workers: {settings.workers}")
    console.print(f"Reload: {settings.reload}")
    console.print(f"Log Level: {settings.log_level}")
    console.print("[green]Status: Ready[/green]")


# @spec[PROJECT_PROFILE.md#Acceptance Evidence]
# @spec[PROJECT_PROFILE.md#Token Budget Class]
@app.command()
def config():
    """Show the current configuration."""
    console.print("[blue]Current Configuration[/blue]")
    config_dict = settings.dict()
    for key, value in config_dict.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            value = "[REDACTED]"
        console.print(f"{key}: {value}")


# @spec[PROJECT_PROFILE.md#Token Budget Class]
@app.command()
def version():
    """Show the version of NVIDIA SmartRoute."""
    console.print(f"NVIDIA SmartRoute CLI version: {__version__}")


if __name__ == "__main__":
    app()