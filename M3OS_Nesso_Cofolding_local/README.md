# M3OS Nesso Cofolding MCP

M3OS adapter for host-local, resident
[Nesso](https://github.com/recursionpharma/nesso) protein-ligand affinity
inference. Workers keep the model resident and can distribute candidate batches
across selected GPUs.

## MCP tools

- `nesso_predict_by_cofolding`: score supplied SMILES against a protein sequence
  or FASTA file.
- `nesso_filter_by_cofolding`: score and rank a candidate CSV, returning a
  chainable output CSV path.

Exactly one of `protein_sequence` and `protein_fasta_path` is required. A
candidate CSV can be supplied explicitly or resolved from a REINVENT generation
identifier.

## Source and model setup

Nesso source, model files, ESM weights, CCD data, and caches are not included.
Clone the official source into the path expected by the setup script:

```bash
cd M3OS_Nesso_Cofolding_local
mkdir -p vendor
git clone https://github.com/recursionpharma/nesso.git vendor/nesso
git -C vendor/nesso checkout 6c72f66720d9d3447fd73c515cda963e39128b1f
./setup_local_nesso.sh
```

The pinned commit records the upstream `main` revision used when this public
release was prepared. Nesso downloads its official model assets on first use.
Set `NESSO_CACHE_DIR` to place large caches outside the repository.

By default, the setup script creates a Python 3.11 inference environment at
`.nesso_venv`. For an existing CUDA-enabled Python, set
`NESSO_BASE_PYTHON=/path/to/python`; the environment is then created with
system-site packages. Set `NESSO_INSTALL_KERNELS=1` to request Nesso's optional
CUDA kernels.

## Run

```bash
./open_mcp_server.sh
```

The default endpoint is `http://127.0.0.1:8051/mcp`. The MCP wrapper uses
`.venv`; model inference uses `.nesso_venv`.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `NESSO_LOCAL_GPU_IDS` | visible GPUs | Comma-separated physical GPU IDs |
| `NESSO_LOCAL_GPU_ID` | unset | Force one physical GPU |
| `NESSO_MCP_MAX_GPUS` | all selected | Maximum resident workers |
| `NESSO_MCP_BATCH_SIZE` | `32` | Molecules assigned per worker job |
| `NESSO_DATALOADER_BATCH_SIZE` | `32` | Per-GPU inference batch |
| `NESSO_PREPROCESS_WORKERS` | `4` | CPU preprocessing workers |
| `NESSO_LOCAL_PYTHON` | `.nesso_venv/bin/python` | Inference interpreter |
| `NESSO_CACHE_DIR` | `nesso_cache` | Model and feature cache |

Stop the MCP endpoint while retaining resident workers with
`./close_mcp_server.sh`. Release worker GPU memory with
`./stop_local_worker.sh`.

## Tests and reproducibility

```bash
uv run --frozen python -m unittest -v \
  test_local_backend.py test_nesso_activity_tools.py
```

Model-backed validation additionally requires downloaded weights and compatible
GPU dependencies. Record the Nesso commit, model-cache revision, PyTorch/CUDA
stack, GPU selection, and batch settings. Conformer preprocessing can be
stochastic, so identical SMILES are not guaranteed to be bitwise reproducible
after cache invalidation.

## Citation

Cite the M3OS paper and the Nesso publication/repository corresponding to the
model revision used in the experiment. Nesso's upstream source declares the
Apache-2.0 license; model assets may have additional terms.

