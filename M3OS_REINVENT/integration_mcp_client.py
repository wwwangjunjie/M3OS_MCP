#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REINVENT MCP Server Test
HTTP MCP test client for reinvent_mcp_server.py
"""

import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# =============================================================================
# MCP CLIENT
# =============================================================================

class ReinventMCPClient:

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8041/mcp"
    ):
        self.server_url = server_url

        self.session = None
        self._ctx = None

    async def connect_to_server(self):

        self._ctx = streamablehttp_client(
            self.server_url
        )

        read, write, _ = await self._ctx.__aenter__()

        self.session = ClientSession(read, write)

        await self.session.__aenter__()

        await self.session.initialize()

        print(f"Connected to MCP server: {self.server_url}")

    async def call_tool(self, tool_name: str, arguments: dict):

        result = await self.session.call_tool(
            tool_name,
            arguments
        )

        return result

    async def cleanup(self):

        if self.session:
            await self.session.__aexit__(None, None, None)

        if self._ctx:
            await self._ctx.__aexit__(None, None, None)

        print("MCP client cleaned up")


# =============================================================================
# HEALTH CHECK
# =============================================================================

async def test_health_check():

    client = ReinventMCPClient()

    await client.connect_to_server()

    result = await client.call_tool(
        "health_check",
        {}
    )

    print("\n=== HEALTH CHECK ===")
    print(result)

    await client.cleanup()


# =============================================================================
# SETUP GENERATION
# =============================================================================

async def test_setup_generation():

    client = ReinventMCPClient()

    await client.connect_to_server()

    result = await client.call_tool(
        "setup_generation",
        {
            "model_type": "Mol2Mol"
        }
    )

    print("\n=== SETUP GENERATION ===")
    print(result)

    await client.cleanup()


# =============================================================================
# PREPARE MOL2MOL INPUT
# =============================================================================

async def test_prepare_mol2mol_input():

    client = ReinventMCPClient()

    await client.connect_to_server()

    smiles = (
        "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)"
        "c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)"
        "c(Oc4cnc5[nH]ccc5c4)c3)CC2)"
        "=C(c2ccc(Cl)cc2)C1"
    )

    result = await client.call_tool(
        "prepare_mol2mol_input",
        {
            "smiles": smiles
        }
    )

    print("\n=== PREPARE MOL2MOL INPUT ===")
    print(result)

    await client.cleanup()


# =============================================================================
# RUN GENERATION
# =============================================================================

async def test_run_generation():

    client = ReinventMCPClient()

    await client.connect_to_server()

    result = await client.call_tool(
        "run_generation",
        {
            "config_file": "configs/Mol2Mol.toml"
        }
    )

    print("\n=== RUN GENERATION ===")
    print(result)

    await client.cleanup()


# =============================================================================
# LIBINVENT SETUP
# =============================================================================

async def test_setup_libinvent():

    client = ReinventMCPClient()

    await client.connect_to_server()

    result = await client.call_tool(
        "setup_libinvent",
        {
            "num_samples": 100,
            "device": "cuda:0"
        }
    )

    print("\n=== SETUP LIBINVENT ===")
    print(result)

    await client.cleanup()


# =============================================================================
# PREPARE SCAFFOLD
# =============================================================================

async def test_prepare_scaffold():

    client = ReinventMCPClient()

    await client.connect_to_server()

    scaffold_smiles = json.dumps([
        "[*]c1ccccc1[*]",
        "[*]c1ncncc1"
    ])

    result = await client.call_tool(
        "prepare_scaffold",
        {
            "scaffold_smiles_list_str": scaffold_smiles,
            "rand_str": "abcdefghij"
        }
    )

    print("\n=== PREPARE SCAFFOLD ===")
    print(result)

    await client.cleanup()


# =============================================================================
# LINKINVENT
# =============================================================================

async def test_complete_reinvent_workflow():

    print("\n" + "=" * 80)
    print("COMPLETE REINVENT WORKFLOW")
    print("=" * 80)

    client = ReinventMCPClient()

    await client.connect_to_server()

    # ==========================================
    # STEP 1
    # ==========================================

    print("\n[STEP 1] setup_generation")

    setup_result = await client.call_tool(
        "setup_generation_Mol2Mol_LinkInvent",
        {
            "model_type": "Reinvent",
            "num_samples": 50,
            "device": "cuda:0"
        }
    )

    print(setup_result)

    config_file = setup_result.content[0].text
    rand_str = setup_result.content[1].text

    print(f"config_file: {config_file}")
    print(f"rand_str: {rand_str}")

    # ==========================================
    # STEP 2
    # ==========================================

    print("\n[STEP 2] run_generation")

    generation_result = await client.call_tool(
        "run_generation",
        {
            "config_file": config_file
        }
    )

    print(generation_result)

    print("\n" + "=" * 80)
    print("WORKFLOW FINISHED")
    print("=" * 80)

    await client.cleanup()


async def test_complete_mol2mol_workflow():

    print("\n" + "=" * 80)
    print("COMPLETE MOL2MOL WORKFLOW")
    print("=" * 80)

    client = ReinventMCPClient()

    await client.connect_to_server()

    # ==========================================
    # STEP 1
    # ==========================================

    print("\n[STEP 1] setup_generation")

    setup_result = await client.call_tool(
        "setup_generation_Mol2Mol_LinkInvent",
        {
            "model_type": "Mol2Mol",
            "num_samples": 50,
            "device": "cuda:0"
        }
    )

    print(setup_result)

    config_file = setup_result.content[0].text
    rand_str = setup_result.content[1].text

    print(f"config_file: {config_file}")
    print(f"rand_str: {rand_str}")

    # ==========================================
    # STEP 2
    # ==========================================

    print("\n[STEP 2] prepare_mol2mol_input")

    smiles = [
        "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(Oc4cnc5[nH]ccc5c4)c3)CC2)=C(c2ccc(Cl)cc2)C1",
    ]
    file_path = os.path.join(os.path.dirname(config_file), "input.smi")
    input_result = await client.call_tool(
        "prepare_smi_input",
        {
            "smiles": smiles,
            "file_path": file_path,
        }
    )

    print(input_result)

    # ==========================================
    # STEP 3
    # ==========================================

    print("\n[STEP 3] run_generation")

    generation_result = await client.call_tool(
        "run_generation",
        {
            "config_file": config_file
        }
    )

    print(generation_result)

    print("\n" + "=" * 80)
    print("WORKFLOW FINISHED")
    print("=" * 80)

    await client.cleanup()

# =============================================================================
# COMPLETE LIBINVENT WORKFLOW
# =============================================================================

async def test_complete_libinvent_workflow():

    print("\n" + "=" * 80)
    print("COMPLETE LIBINVENT WORKFLOW")
    print("=" * 80)

    client = ReinventMCPClient()

    await client.connect_to_server()

    # ==========================================
    # STEP 1
    # ==========================================

    print("\n[STEP 1] setup_libinvent")

    setup_result = await client.call_tool(
        "setup_libinvent",
        {
            "num_samples": 50,
            "device": "cuda:0"
        }
    )

    print(setup_result)

    # FastMCP result parsing

    config_file = setup_result.content[0].text
    rand_str = setup_result.content[1].text

    print(f"config_file: {config_file}")
    print(f"rand_str: {rand_str}")

    # ==========================================
    # STEP 2
    # ==========================================

    print("\n[STEP 2] prepare_scaffold")

    # scaffold_smiles = json.dumps([
    #     "[*:1]c1ccccc1[*:2]"
    # ])

    scaffold_smiles = json.dumps([
        "*c1ccccc1*"
    ])

    scaffold_result = await client.call_tool(
        "prepare_scaffold",
        {
            "scaffold_smiles_list_str": scaffold_smiles,
            "rand_str": rand_str
        }
    )

    print(scaffold_result)

    # ==========================================
    # STEP 3
    # ==========================================

    print("\n[STEP 3] run_generation")

    generation_result = await client.call_tool(
        "run_generation",
        {
            "config_file": config_file
        }
    )

    print(generation_result)

    print("\n" + "=" * 80)
    print("WORKFLOW FINISHED")
    print("=" * 80)

    await client.cleanup()


async def test_complete_linkinvent_workflow():

    print("\n" + "=" * 80)
    print("COMPLETE LINKINVENT WORKFLOW")
    print("=" * 80)

    client = ReinventMCPClient()

    await client.connect_to_server()

    # ==========================================
    # STEP 1
    # ==========================================

    print("\n[STEP 1] setup_linkinvent")

    setup_result = await client.call_tool(
        "setup_generation_Mol2Mol_LinkInvent",
        {
            "model_type": "LinkInvent",
            "num_samples": 50,
            "device": "cuda:0"
        }
    )


    print(setup_result)

    config_file = setup_result.content[0].text
    rand_str = setup_result.content[1].text

    print(f"config_file: {config_file}")
    print(f"rand_str: {rand_str}")

    # ==========================================
    # STEP 2
    # ==========================================

    print("\n[STEP 2] prepare_linkinvent_input")

    smiles = [
        "[*]CC1=CC=CC=C1.[*]N1CCCCC1",
    ]

    file_path = os.path.join(os.path.dirname(config_file), "input.smi")
    input_result = await client.call_tool(
        "prepare_smi_input",
        {
            "smiles": smiles,
            "file_path": file_path,
        }
    )

    print(input_result)

    # ==========================================
    # STEP 3
    # ==========================================

    print("\n[STEP 3] run_generation")

    generation_result = await client.call_tool(
        "run_generation",
        {
            "config_file": config_file
        }
    )

    print(generation_result)

    print("\n" + "=" * 80)
    print("WORKFLOW FINISHED")
    print("=" * 80)

    await client.cleanup()


# =============================================================================
# SHOW MENU
# =============================================================================

async def show_menu():

    tools = {
        "1": ("health_check", test_health_check),
        "2": ("setup_generation", test_setup_generation),
        "3": ("prepare_mol2mol_input", test_prepare_mol2mol_input),
        "4": ("run_generation", test_run_generation),
        "5": ("setup_libinvent", test_setup_libinvent),
        "6": ("prepare_scaffold", test_prepare_scaffold),
        "7": ("complete_reinvent_workflow",test_complete_reinvent_workflow),
        "8": ("complete_libinvent_workflow", test_complete_libinvent_workflow),
        "9": ("complete_mol2mol_workflow",test_complete_mol2mol_workflow),
        "10": ("complete_linkinvent_workflow",test_complete_linkinvent_workflow),
    }

    print("\n=== REINVENT MCP TEST MENU ===")

    for k, (name, _) in tools.items():
        print(f"{k}. {name}")

    print("0. exit")

    choice = input("\nSelect: ").strip()

    if choice == "0":

        return

    elif choice in tools:

        name, func = tools[choice]

        print(f"\nRunning test: {name}")

        await func()

    else:

        print("Invalid choice")


async def test_list_tools():

    client = ReinventMCPClient()

    await client.connect_to_server()

    tools = await client.session.list_tools()

    print("\n=== MCP TOOLS ===")

    for tool in tools.tools:
        print(tool.name)

    await client.cleanup()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # interactive menu
    # asyncio.run(show_menu())

    # direct tests
    # asyncio.run(test_list_tools())

    # asyncio.run(test_setup_generation())

    # asyncio.run(test_run_generation())

    # asyncio.run(test_setup_libinvent())

    # asyncio.run(test_list_tools())

    asyncio.run(test_complete_reinvent_workflow())
    # asyncio.run(test_complete_libinvent_workflow())
    # asyncio.run(test_complete_mol2mol_workflow())
    # asyncio.run(test_complete_linkinvent_workflow())
