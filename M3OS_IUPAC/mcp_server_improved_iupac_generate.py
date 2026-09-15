import os
import signal
import sys
import logging

from mcp.server import FastMCP
from iupac_generate import get_iupac_names


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_mcp_port(default_port: int) -> int:
    return int(os.environ.get("MCP_SERVER_PORT", default_port))


app = FastMCP('iupac_name_generate_server', host="0.0.0.0", port=get_mcp_port(8020))


@app.tool()
async def generate_iupac_name(smiles_list: list[str]) -> dict[str, str | None]:
    """
    Generate the IUPAC names for multiple molecules based on their SMILES.

    Args:
        smiles_list (list): List of SMILES representations of molecules.
            - Example: ['Cc1ccc(C(=O)[O-])c(Cl)c1', 'CC(=O)Nc1ccc(C(=O)Nc2ccncc2)cc1']
            
    Returns:
        dict: A dictionary where the key is the SMILES string and the value is the corresponding IUPAC name.
            - Example: {'Cc1ccc(C(=O)[O-])c(Cl)c1': '2-chloro-4-methylbenzoate',
                       'CC(=O)Nc1ccc(C(=O)Nc2ccncc2)cc1': '4-acetamido-N-pyridin-4-ylbenzamide'}
            Returns None for any SMILES that cannot be processed.
    """
    return get_iupac_names(smiles_list=smiles_list)


def signal_handler(_signum, _frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutting down m3os IUPAC MCP server...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Starting m3os IUPAC MCP server with FastMCP...")
        logger.info("Available tools: 1 tools across 1 categories")
        logger.info("Available prompts: 0 expert prompt templates")
        logger.info("Available resources: 0 information resources")

        app.run(transport="streamable-http")
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)
