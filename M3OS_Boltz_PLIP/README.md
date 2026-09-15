# M3OS Boltz/PLIP MCP

M3OS adapter for two complementary structure tasks:

- [Boltz](https://github.com/jwohlwend/boltz) predicts a protein-ligand complex.
- [PLIP](https://github.com/pharmai/plip) profiles non-covalent interactions in
  a supplied PDB, CIF, or mmCIF complex.

M3OS uses these tools for structural context, not as a replacement for
experimental affinity measurements.

## MCP tools

- `generate_protein_ligand_complex`: run Boltz from a protein sequence and
  ligand SMILES, returning predicted structure and related output paths.
- `analyze_protein_ligand_interactions`: analyze an existing complex with PLIP
  and return interaction categories plus a JSON output path.

The tools are independent. A caller can pass the structure produced by the
first tool to the second, or analyze a pre-existing complex directly.

## Installation

The wrapper requires Python 3.12 or newer. Boltz is installed by the M3OS
manifest. PLIP source is deliberately not included and must be obtained from
its official repository:

```bash
cd M3OS_Boltz_PLIP
uv sync --frozen
git clone --branch v3.0.1 --depth 1 https://github.com/pharmai/plip.git plip_source
uv sync --project plip_source
```

The launcher expects `plip_source/.venv/bin/python`. To use a different PLIP
installation, set `PLIP_PYTHON`. Boltz downloads official model assets into
`boltz_data/` on first use; this cache is ignored by Git.

PLIP requires Open Babel and may require system packages described in the
upstream installation guide.

## Run

```bash
./open_mcp_server.sh
```

The default endpoint is `http://127.0.0.1:8040/mcp`.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `BOLTZ_MAX_CONCURRENT_REQUESTS` | `1` | Concurrent Boltz processes |
| `PLIP_MAX_CONCURRENT_REQUESTS` | `4` | Concurrent PLIP processes |
| `BOLTZ_SUBPROCESS_TIMEOUT_SECONDS` | `7200` | Boltz execution timeout |
| `PLIP_SUBPROCESS_TIMEOUT_SECONDS` | `300` | PLIP execution timeout |
| `BOLTZ_PYTHON` | `.venv/bin/python` | Boltz Python interpreter |
| `PLIP_PYTHON` | `plip_source/.venv/bin/python` | PLIP Python interpreter |

Outputs are written below `boltz_output/`; model files and caches live below
`boltz_data/`. Both are excluded from version control.

## Tests and citation

```bash
uv run --frozen pytest test_split_tools.py
```

Model-backed checks require their upstream installations and assets. Cite the
M3OS paper, the applicable Boltz paper, and the PLIP paper listed in each
official repository. PLIP is GPL-2.0; review upstream license obligations for
your deployment.
