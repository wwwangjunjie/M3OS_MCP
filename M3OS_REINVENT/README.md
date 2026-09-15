# M3OS REINVENT MCP

M3OS adapter for molecular generation with
[REINVENT4](https://github.com/MolecularAI/REINVENT4). The adapter prepares
sampling TOML files, writes SMILES inputs, invokes a separate REINVENT runtime,
and emits candidate CSV files for downstream filtering.

## MCP tools

- `setup_generation_Mol2Mol_LinkInvent`: create a Mol2Mol or LinkInvent
  sampling configuration.
- `prepare_smi_input`: write a generation input file.
- `run_generation`: execute REINVENT and return the generated CSV metadata.
- `generate_similarity_constrained_mol2mol`: sample, deduplicate, and filter
  local analogues by an RDKit Morgan-fingerprint Tanimoto threshold.
- `setup_libinvent` and `prepare_scaffold`: prepare LibInvent R-group jobs.
- `health_check`: report adapter and upstream runtime status.

## Source and prior setup

REINVENT source, environments, and prior/model files are intentionally not
included. Install the tested upstream release in the expected directory:

```bash
cd M3OS_REINVENT
git lfs install
git clone --branch v4.5.11 --depth 1 \
  https://github.com/MolecularAI/REINVENT4.git REINVENT4
git -C REINVENT4 lfs pull
uv sync --project REINVENT4
uv sync --frozen
```

Follow REINVENT's official installation documentation if your CUDA, ROCm, CPU,
or licensed optional-component setup requires a different command. Confirm that
the priors required by your chosen mode exist below `REINVENT4/priors/`.

For an upstream installation at another location, set both:

```bash
export REINVENT_ROOT=/path/to/REINVENT4
export REINVENT_PYTHON=/path/to/REINVENT4/.venv/bin/python
```

## Run

```bash
./open_mcp_server.sh
```

The default endpoint is `http://127.0.0.1:8041/mcp`. Candidate job files and
outputs are written below `tmp/reinvent/` and `tmp/pool/`; both are runtime data
and are excluded from version control.

The similarity-constrained path filters with binary Morgan fingerprints
(radius 2, 2048 bits, chirality disabled) and records the resulting score as
`SMDD_Tanimoto`. Generation remains stochastic unless all relevant upstream
randomness and hardware settings are controlled.

## Tests and reproducibility

```bash
uv run --frozen pytest test_generation_result.py
```

After starting the server, run `uv run integration_mcp_client.py` for live MCP
checks. Generation checks require the upstream environment and matching priors.
Record the REINVENT release, prior filenames and hashes, sampling TOML, random
seed, decoding strategy, temperature, and device for every paper result.

## Citation

Cite the M3OS paper and the REINVENT4 paper listed in the
[official REINVENT4 repository](https://github.com/MolecularAI/REINVENT4).
REINVENT source and priors remain subject to their upstream licenses.
