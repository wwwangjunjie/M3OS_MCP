# M3OS MCP Services

This repository contains the public Model Context Protocol (MCP) adapters used
by M3OS. Each directory is an independent service with its own environment,
port, runtime files, and reproducibility instructions.

The repository intentionally contains only M3OS integration code. Third-party
source trees, model checkpoints, prior files, caches, generated structures,
virtual environments, and private services are not redistributed.

## Available services

| Directory | Port | Capability | External dependency |
| --- | ---: | --- | --- |
| [`M3OS_ADMET_AI`](M3OS_ADMET_AI/) | 8000 | ADMET prediction and filtering | ADMET-AI package/models |
| [`M3OS_IUPAC`](M3OS_IUPAC/) | 8020 | SMILES-to-IUPAC conversion | STOUT package/models |
| [`M3OS_Boltz_PLIP`](M3OS_Boltz_PLIP/) | 8040 | Complex prediction and interaction profiling | Boltz package/models and PLIP source |
| [`M3OS_REINVENT`](M3OS_REINVENT/) | 8041 | De novo, Mol2Mol, LinkInvent, and LibInvent generation | REINVENT4 source and priors |
| [`M3OS_Nesso_Cofolding_local`](M3OS_Nesso_Cofolding_local/) | 8051 | Protein-ligand affinity scoring | Nesso source and model assets |
| [`M3OS_TamGen`](M3OS_TamGen/) | 8057 | Target-aware molecular generation | TamGen source and checkpoints |

`M3OS_fast_medchem_request` and `M3OS_MMP` are private components and are not
part of this release.

## Design

```text
M3OS agent runtime
      |
      | streamable HTTP MCP
      v
service-specific M3OS adapter
      |
      +--> separately installed upstream package/source
      +--> separately downloaded model assets
      +--> ignored runtime cache and outputs
```

Keeping these layers separate avoids republishing third-party code or weights,
makes upstream licenses explicit, and lets users install only the services
required by an experiment.

## Installation policy

Do not create one shared environment for the entire repository. Enter each
service directory and follow its README. For a lightweight service, the common
pattern is:

```bash
cd M3OS_ADMET_AI
uv sync --frozen
./open_mcp_server.sh
```

Model-backed services add an upstream source/checkpoint step before starting.
Their README records the official repository, a tested tag or commit where
available, the expected local directory layout, and environment variables.

The checked-in `pyproject.toml` and `uv.lock` files describe M3OS wrapper
environments. Upstream projects retain their own manifests and installation
procedures after being cloned locally.

## Connecting M3OS

Start only the services needed for an experiment, then configure their URLs in
the main M3OS `.env` file. For example:

```dotenv
MCP_SERVER_ADMET_AI_URL=http://127.0.0.1:8000/mcp
MCP_SERVER_IUPAC_GEN_URL=http://127.0.0.1:8020/mcp
MCP_SERVER_BOLTZ_URL=http://127.0.0.1:8040/mcp
MCP_SERVER_REINVENT_URL=http://127.0.0.1:8041/mcp
MCP_SERVER_NESSO_URL=http://127.0.0.1:8051/mcp
```

TamGen is a standalone optional MCP endpoints. A custom M3OS
deployment may register them with its MCP client when those experimental paths
are enabled.

## Data and security

MCP services accept local paths and may write derived structures, CSV files,
SQLite caches, logs, and model caches. These artifacts are excluded by
`.gitignore`. Review uploaded structures and sequences for disclosure constraints
before sending them to external APIs or model-download services.

The launchers bind to loopback by default where supported. Add authentication,
TLS, and network isolation before exposing an endpoint beyond a trusted host.


