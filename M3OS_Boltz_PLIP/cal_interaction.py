import sys
import json
import argparse
import tempfile
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, PDBIO

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "plip_source"))

from plip.structure.preparation import PDBComplex
from plip.exchange.report import BindingSiteReport


def cif_to_pdb_with_lig_resname(cif_file, pdb_out):
    """Convert CIF to PDB and rename all HETATM residues to LIG."""
    suffix = Path(cif_file).suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        print(f"[ERROR] Unsupported file type: {cif_file}")
        return None

    try:
        structure = parser.get_structure("struct", cif_file)
    except Exception as e:
        print(f"[ERROR] Failed to parse {cif_file}: {e}")
        return None

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # HETATM
                    residue.resname = "LIG"

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_out)
    print(f"[OK] CIF -> PDB: {pdb_out}")
    return pdb_out


def get_interaction(pdb_file):
    """Extract PLIP interaction information from a PDB file."""
    complex_structure = PDBComplex()
    complex_structure.load_pdb(pdb_file)
    complex_structure.analyze()

    interaction_types = {
        "hbond_info": "Hydrogen Bonds",
        "hydrophobic_info": "Hydrophobic Contacts",
        "waterbridge_info": "Water Bridges",
        "saltbridge_info": "Salt Bridges",
        "pistacking_info": "Pi Stackings",
        "pication_info": "Pi Cations",
        "halogen_info": "Halogens",
        "metal_info": "Metal Complexes",
    }
    interaction_results = {}

    for ligand_id, interaction in complex_structure.interaction_sets.items():
        report = BindingSiteReport(interaction)
        for attr, title in interaction_types.items():
            details = getattr(report, attr)
            interaction_results[title] = [len(details), details]

    return interaction_results


def analyze_structure(structure_file):
    """Run PLIP for a PDB file, converting CIF/mmCIF inputs when needed."""
    source = Path(structure_file).resolve()
    suffix = source.suffix.lower()
    if suffix == ".pdb":
        return get_interaction(str(source))
    if suffix not in {".cif", ".mmcif"}:
        raise ValueError(f"Unsupported structure format: {source.suffix}")

    with tempfile.TemporaryDirectory(prefix="plip_cif_") as temp_dir:
        converted_pdb = Path(temp_dir) / f"{source.stem}.pdb"
        converted = cif_to_pdb_with_lig_resname(
            str(source),
            str(converted_pdb),
        )
        if converted is None:
            raise RuntimeError(f"Failed to convert CIF structure: {source}")
        return get_interaction(converted)


def main():
    parser = argparse.ArgumentParser(
        description="PLIP interaction analysis from PDB files."
    )
    parser.add_argument("--input", "-i", required=True, help="Input CIF or PDB file")
    parser.add_argument("--output", "-o", required=True, help="Output plip json")
    args = parser.parse_args()

    pdb_file = args.input
    output_json = Path(args.output)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    results = analyze_structure(pdb_file)
    with output_json.open("w") as f:
        json.dump(results, f, indent=4)

    print(f"Interaction results saved to {output_json}")


if __name__ == "__main__":
    main()
