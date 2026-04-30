# OpenTela

[![GitHub Repo](https://img.shields.io/badge/github-eth--easl%2FOpenTela-black?logo=github)](https://github.com/eth-easl/OpenTela) ![CI Workflow](https://github.com/eth-easl/OpenTela/actions/workflows/ci.yml/badge.svg) [![License](https://img.shields.io/github/license/eth-easl/OpenTela)](LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/eth-easl/OpenTela) [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/pAsWxTYttP)

**OpenTela** (Aka: OpenFabric) is a distributed computing platform designed to orchestrate computing resources across a decentralized network. It leverages peer-to-peer networking, CRDT-based state management to create a resilient and scalable network of computing resources. It is used to power the [serving system at SwissAI Initiative](https://serving.swissai.cscs.ch).

Tela is the latin word for "Fabric", which refers to the interconnected network of computing resources that OpenTela manages.

## Latest Updates

*   **[2026/02]** 💡 **How SwissAI Leverages OpenTela**: We wrote a case study on how SwissAI uses OpenTela to orchestrate their distributed GPU nodes for scalable model serving. [Read more](docs/posts/swissai.md).

## Features

- **Decentralized Orchestration**: OpenTela eliminates the need for a central coordinator by using a gossip-based P2P network. It utilizes a Conflict-free Replicated Data Type (CRDT) registry to manage service discovery, health monitoring, and routing across distributed nodes. This architecture allows the system to remain operational and maintain a global view of resources even during network partitions.

- **Non-Invasive HPC Integration**: Designed specifically for the constraints of supercomputing environments, the system operates entirely as a user-space overlay. It bridges the gap between batch schedulers (like Slurm) and interactive serving engines (like vLLM or SGLang) without requiring root privileges or kernel modifications. This allows researchers to spin up "cloud-like" serving clusters using standard permissions.

- **Robust Fault Tolerance and Elasticity**: OpenTela is built for high-churn environments where resources are often volatile or preemptible (e.g., [scavenger queues](https://docs.icer.msu.edu/Scavenger_Queue/), [preemptible cloud instances](https://docs.cloud.google.com/compute/docs/instances/preemptible) or [slurm preemption](https://slurm.schedmd.com/preempt.html)). It utilizes peer-to-peer heartbeats to detect node failures within seconds, automatically marking failed nodes as "LEFT" and rerouting traffic to healthy replicas without service interruption.

## Adoption

- OpenTela is used to power [SwissAI Serving](https://serving.swissai.cscs.ch/). It acts as the decentralized orchestration layer, routing inference requests to distributed GPU nodes while managing state, metrics, and peer discovery to ensure resilient and scalable model serving.

## Documentation

### Getting Started
- [Installation](docs/tutorial/installation) — Download and install OpenTela
- [Spin Up LLM Serving](docs/tutorial/spinup) — Set up multi-LLM serving cluster
- [Request Routing](docs/tutorial/routing) — Understand how requests are routed
- [Wallet & Ownership](docs/tutorial/owner) — Manage Solana wallets and node identity
- [Solana Settlement](docs/tutorial/settlement) — Configure automated usage billing
- [Docker Serving](docs/tutorial/docker-serving) — Use Docker containers for LLM serving
- [Glossary](docs/tutorial/glossary) — Key terms and concepts

### Advanced Topics
- [CRDT Internals](docs/advanced/crdt-internals) — How CRDT synchronization works
- [CRDT Tombstones](docs/advanced/crdt-tombstones) — Node departure handling
- [Security Hardening](docs/advanced/security) — Build attestation, trust, and access control
- [Performance Benchmark](docs/advanced/performance-optimization) — Proxy latency measurements
- [Large-Scale Simulation](docs/advanced/benchmark) — Run 100+ node simulations

### Extensions
- [Fleet Manager](docs/extensions/fleet-manager) — Deploy to SLURM clusters with otela-fleet

## Contributing

Contributions are welcome! Please follow the code of conduct and submit pull requests for any enhancements or bug fixes.

## License

This project is licensed under the Apache v2 License - see the [LICENSE](LICENSE) file for details.
