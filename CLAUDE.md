# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenTela (formerly OpenFabric) is a decentralized distributed computing platform that orchestrates computing resources across a peer-to-peer network. It uses libp2p networking, CRDT-based state management, and gossip protocols. Primary use case: distributed GPU node orchestration for LLM serving (SwissAI Initiative).

## Build & Development Commands

All commands run from the `src/` directory:

```bash
make build                                    # Build binary (output: src/build/entry)
make build-release                            # Optimized binaries for amd64 & arm64
make build-debug                              # Build with debug symbols
make test                                     # Run all tests with coverage
make test TEST_PKGS="./internal/protocol/..."  # Run tests for a specific package
make test VERBOSE=1                           # Verbose test output
make lint                                     # Run golangci-lint (v2, config in .golangci.yml)
make check                                    # Run both tests and lint
```

Tests require `CGO_ENABLED=1` (set automatically by Makefile) and use `-race` flag. Coverage output goes to `build/coverage.xml` and `build/coverage.txt`.

## Architecture

### Network Topology

Nodes form a P2P mesh using libp2p. A **head node (dispatcher)** receives client HTTP requests and forwards them over libp2p to **worker nodes** running local services (e.g., vLLM).

### Request Routing Flow

1. Client sends request to head node at `/v1/service/:service/*path`
2. Head node queries the distributed node table for peers matching the requested **identity group** (e.g., `model=Qwen/Qwen3-8B`)
3. Three-tier matching with fallback (`X-Otela-Fallback` header): exact match → wildcard (`model=*`) → catch-all (`all`)
4. Request forwarded over libp2p to selected worker, which proxies to its local service

### Key Source Directories (`src/`)

- **`entry/`** — CLI entry point using Cobra. `cmd/root.go` initializes config (Viper), `cmd/start.go` starts the node.
- **`internal/protocol/`** — P2P networking core. `host.go` sets up libp2p host. `crdt.go` initializes CRDT datastore (Badger-backed). `node_table.go` tracks peers and services. `registrar.go` handles service registration.
- **`internal/server/`** — HTTP server (Gin). `server.go` sets up endpoints. `proxy_handler.go` contains routing logic (`GlobalServiceForward`, `P2PForwardHandler`). `crdt_handler.go` exposes CRDT APIs.
- **`internal/protocol/go-ds-crdt/`** — Forked/modified CRDT datastore implementation with tombstone management and compaction.
- **`internal/common/`** — Shared utilities, constants, logging, process management.
- **`internal/wallet/`** and **`internal/solana/`** — Solana wallet management and SPL token verification.
- **`internal/platform/`** — Hardware detection (NVIDIA GPU via nvidia-smi, Slurm environment).

### Key Patterns

- **CRDT consensus**: Badger DB + tombstone-based deletion (24h retention, hourly compaction). No central coordinator.
- **Gossip protocol**: libp2p PubSub for broadcasting, Kademlia DHT for peer discovery, 20s ping interval.
- **Config**: Viper with YAML config at `$HOME/.config/otela/cfg.yaml`, env vars prefixed `OF_`.
- **CGO_ENABLED=0** for builds (no C deps), but **CGO_ENABLED=1** for tests (race detector).

## Usage Tracking and Billing

OpenTela includes an opt-in dual-attestation billing system for tracking resource consumption across the decentralized network.

### Configuration

Enable billing in `cfg.yaml`:

```yaml
billing:
  enabled: true
  value_threshold: 10000000
  max_interval_minutes: 60
  dispute_threshold_pct: 10
```

Or via environment variable: `OF_BILLING_ENABLED=true`

### Service Integration

Services report usage via HTTP response headers:

```
X-Usage-Tokens: 1234
X-Usage-GPU-Ms: 5000
```

### Data Flow

1. Head and worker nodes independently extract usage from response headers
2. Records aggregate locally until threshold triggers (value or time)
3. Aggregates shared via CRDT for reconciliation
4. Resolved usage submitted to Solana for settlement

### Components

- `internal/usage/` - Core usage tracking (types, extractor, store, reconciler, aggregator)
- `internal/solana/settlement.go` - Blockchain settlement (skeleton)
- `internal/server/proxy_handler.go` - Integration point for tracking

See `docs/proposals/2026-03-08-usage-tracking-design.md` for full design details.

## CI

- **Tests**: `make test VERBOSE=1` in `src/` (GitHub Actions)
- **Lint**: golangci-lint v2.6 with 5m timeout
- **Release**: Triggered by `v*` tags; builds binaries and Docker images pushed to `ghcr.io`
