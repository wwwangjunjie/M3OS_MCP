# M3OS IUPAC MCP

M3OS adapter for converting SMILES strings to IUPAC names with
[STOUT](https://github.com/Kohulan/Smiles-TO-iUpac-Translator). The service
exposes one streamable HTTP MCP tool.

## MCP tool

`generate_iupac_name` accepts a list of SMILES and returns a mapping from each
input to an IUPAC name or `null` when conversion fails.

```json
{
  "smiles_list": ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
}
```

## Installation and run

Requires Python 3.10 or newer. STOUT is installed from the dependency manifest;
no upstream source or model cache is included here.

```bash
cd M3OS_IUPAC
uv sync --frozen
./open_mcp_server.sh
```

The default endpoint is `http://127.0.0.1:8020/mcp`. Use
`MCP_SERVER_PORT=8021 ./open_mcp_server.sh` or pass a port as the first argument
to change it.

Successful conversions are cached in `smiles_iupac_dict.json`. The file is
created at runtime and intentionally excluded from version control; an empty
cache is valid for a fresh installation.

## Reproducibility and citation

Record the STOUT package/model version together with the service lock file.
Name generation may change when upstream models change. Cite the M3OS paper and
the STOUT publication referenced by the
[official project](https://github.com/Kohulan/Smiles-TO-iUpac-Translator).

