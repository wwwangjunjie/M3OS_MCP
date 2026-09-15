# M3OS ADMET-AI MCP

M3OS adapter for property prediction and constraint-based candidate filtering
with [ADMET-AI](https://github.com/swansonk14/admet_ai). The model is loaded on
first use and retained by the server process.

## MCP tools

- `admet_predict_by_admetai`: predict selected properties for supplied SMILES.
- `admet_filter_by_admetai`: rank a candidate CSV and write a chainable output
  CSV.
- `admet_filter_by_task_constraints`: apply structured ADMET objectives and
  RDKit-computable molecular constraints to the complete candidate set.

Supported RDKit-derived fields include molecular weight, logP, hydrogen-bond
donors/acceptors, Lipinski status, QED, stereocenter count, and TPSA.

## Installation

Requires Python 3.12 or newer. No ADMET-AI source tree or model cache is stored
in this repository; `admet-ai` is declared in `pyproject.toml`.

```bash
cd M3OS_ADMET_AI
uv sync --frozen
```

ADMET-AI obtains its model assets through its normal upstream runtime. Consult
the upstream project for model provenance, supported hardware, and licensing.
The small M3OS property-metadata JSON files are retained because the adapter
uses them at runtime. The upstream supplementary spreadsheet used by
`property_info/extract_prop_info.py` is not redistributed; obtain it from the
ADMET-AI publication before regenerating those JSON files.

## Run

```bash
./open_mcp_server.sh
```

The default endpoint is `http://127.0.0.1:8000/mcp`. Override it with a first
positional port or `MCP_SERVER_PORT`.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `ADMET_MAX_CONCURRENT_REQUESTS` | `1` | Admitted prediction calls |
| `ADMET_NUM_WORKERS` | `0` | Chemprop data-loader workers |
| `ADMET_FINGERPRINT_THREAD_MIN` | `100` | Threshold for threaded fingerprints |
| `M3OS_REINVENT_ROOT` | sibling `M3OS_REINVENT` | Candidate-pool lookup root |

Runtime CSVs, logs, PIDs, caches, and environments are ignored by Git.

## Tests

```bash
uv run --frozen pytest test_admet_runtime.py
```

The benchmark client is available as `benchmark_admet_mcp.py`; start the server
before running it. Report model version, hardware, warm-up policy, batch size,
and concurrency with any performance results.

## Citation

When using this service, cite the M3OS paper and the ADMET-AI publication listed
by the [official ADMET-AI repository](https://github.com/swansonk14/admet_ai).
