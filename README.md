# NVIDIA-SmartRoute-CLI

A CLI application that provides an OpenAI-compatible API gateway for NVIDIA NIM models with dynamic agent scaling and a rich terminal user interface.

## Features

- **Unified OpenAI-Compatible API Gateway**: Listens on port 9000 (`0.0.0.0:9000`) and intercepts `/v1/chat/completions` and `/v1/embeddings` requests
- **Intelligent Request Routing**: Dynamically analyzes incoming requests to determine required capabilities (math, code, vision, etc.) and routes to the best-suited NVIDIA NIM model
- **Dynamic Agent Autoscale Engine**: Automatically spawns specialized sub-agents (Code Writer, Code Tester, etc.) for complex multi-step tasks
- **Rich Terminal User Interface (TUI)**: Real-time dashboard showing metrics, active connections, model performance, and routing logs
- **Secure Design**: No hardcoded secrets, follows security best practices
- **Observable**: Comprehensive logging, metrics, and health checks

## Installation

```bash
pip install nvidia-smartroute
```

## Usage

```bash
# Start the API gateway server
nvidia-smartroute start

# Check server status
nvidia-smartroute status

# View current configuration
nvidia-smartroute config show

# Stop the server
nvidia-smartroute stop
```

## Development Setup

```bash
# Clone repository
git clone https://github.com/joeldg/nvidiarouter.git
cd nvidiarouter

# Install development dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your NVIDIA API credentials
```

## Governance Notice

This project is developed under SpecRegistry governance principles. All implementation work traces to governed specifications from the SpecRegistry.

For more information about the governance framework, see:
- [SPECREGISTRY.md](./SPECREGISTRY.md)
- [AGENTS.md](./AGENTS.md)