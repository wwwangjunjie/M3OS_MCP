# M3OS TamGen MCP

M3OS adapter for target-aware molecule generation with
[TamGen](https://github.com/microsoft/TamGen). The service prepares a protein
pocket from a complex, caches reusable target context, and writes generated
candidates to a CSV with a standard `SMILES` column.

## MCP tools

- `tamgen_prepare_target_context`: resolve a complex, binding pocket, and seed
  molecule into a reusable context.
- `tamgen_generate_target_aware_candidates`: generate candidates from a
  prepared context.
- `tamgen_optimize_lead_from_complex`: prepare context and generate in one call.
- `tamgen_generate_scaffold_conditioned_candidates`: condition generation on
  an explicit scaffold and optionally enforce the substructure.
- `health_check`: report source, checkpoint, and resident-runtime status.

Inputs may include a PDB protein-ligand complex, an SDF/MOL/MOL2 or SMILES seed,
explicit `(chain, residue number)` pocket residues, or a radius-defined pocket.
Generated candidates still require independent validity, property, and activity
screening.

## Source and model setup

TamGen source and checkpoints are not included. Install the upstream source at
the recorded revision and download the official Zenodo model assets:

```bash
cd M3OS_TamGen
git clone https://github.com/microsoft/TamGen.git source
git -C source checkout 9f49e6cee3a861c143600db2f1a9bc10bb1c5279
./download_models.sh
./setup_tamgen.sh
```

`download_models.sh` retrieves and verifies the files from
[Zenodo record 13751391](https://doi.org/10.5281/zenodo.13751391). The expected
layout is:

```text
checkpoints/crossdock_pdb_A10/checkpoint_best.pt
gpt_model/checkpoint_best.pt
gpt_model/dict.txt
source/
```

The setup targets Python 3.10, PyTorch 2.3.0, and CUDA 12.1 wheels. Adjust the
PyTorch installation deliberately for other accelerators and record the change.

## Run

```bash
TAMGEN_GPU_ID=0 ./open_mcp_server.sh
```

The default endpoint is `http://127.0.0.1:8057/mcp`.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `TAMGEN_GPU_ID` | `0` | Physical GPU exposed to the process |
| `TAMGEN_DEVICE` | `cuda:0` | Torch device within the process |
| `TAMGEN_PRELOAD` | `1` | Load the model during startup |
| `TAMGEN_FP16` | `0` | Enable half-precision inference |
| `TAMGEN_DOWNLOAD_DIR` | `/tmp` | Archive download directory |

Contexts and outputs are stored below `workspaces/`; environments, checkpoints,
logs, PIDs, and workspaces are excluded from Git. Stop the service with
`./close_mcp_server.sh`.

## Tests and reproducibility

```bash
.venv/bin/python -m pytest
```

Model-backed tests require the separately installed source and checkpoints.
Record the source commit, checkpoint hashes, pocket definition, seed/scaffold,
sampling parameters, random seeds, GPU, and software stack.

## Citation

Cite the M3OS paper and:

> Wu et al. “TamGen: drug design with target-aware molecule generation through
> a chemical language model.” *Nature Communications* 15, 9360 (2024).
> https://doi.org/10.1038/s41467-024-53632-4

TamGen source is MIT licensed; checkpoints and other assets retain the terms
published by their providers.
