import os
import subprocess
import re
import string
import secrets
from configs.tool_globals import POOL_PATH, REINVENT_JOB_TMP_PATH, REINVENT_PATH, REINVENT_PYTHON
from typing import List


def get_reinvent_job_dir(rand_str: str) -> str:
    job_dir = os.path.join(REINVENT_JOB_TMP_PATH, rand_str)
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def update_reinvent_config(
    model_type: str = "Reinvent",
    device="0",
    num_samples=200,
    rand_str=None,
    mol2mol_model: str = "medium_similarity",
    sample_strategy: str | None = None,
    temperature: float = 1.0,
    random_seed: int | None = None,
):
    """
    Create a REINVENT sampling TOML config file. This returns the path to the updated config file.

    Args:
        model_type (str): The model type to use for updating paths. Default is "Reinvent".
            This can be "Reinvent", "LibInvent", "LinkInvent", or "Mol2Mol".
    """
    if model_type == "LibInvent":
        return update_libinvent_config(
            model_type=model_type,
            device=device,
            num_samples=num_samples,
            rand_str=rand_str,
        )

    mol2mol_models = {
        "high_similarity": os.path.join("priors", "mol2mol_high_similarity.prior"),
        "medium_similarity": os.path.join("priors", "mol2mol_medium_similarity.prior"),
        "similarity": os.path.join("priors", "mol2mol_similarity.prior"),
        "mmp": os.path.join("priors", "mol2mol_mmp.prior"),
        "scaffold": os.path.join("priors", "mol2mol_scaffold.prior"),
        "scaffold_generic": os.path.join("priors", "mol2mol_scaffold_generic.prior"),
    }
    if mol2mol_model not in mol2mol_models:
        choices = ", ".join(sorted(mol2mol_models))
        raise ValueError(
            f"Unsupported Mol2Mol model '{mol2mol_model}'. Supported values: {choices}"
        )

    resolved_strategy = sample_strategy or "beamsearch"
    if resolved_strategy not in {"beamsearch", "multinomial"}:
        raise ValueError("sample_strategy must be 'beamsearch' or 'multinomial'")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    model_configs = {
        "Reinvent": {
            "description": "## Reinvent: de novo sampling",
            "model_file": os.path.join("priors", "reinvent.prior"),
        },
        "Mol2Mol": {
            "description": "## Mol2Mol: find molecules similar to the provided molecules",
            "model_file": mol2mol_models[mol2mol_model],
            "smiles_file": "input.smi",
            "sample_strategy": resolved_strategy,
            "temperature": temperature,
        },
        "LinkInvent": {
            "description": "## LinkInvent: find a linker/scaffold to link two fragments",
            "model_file": os.path.join("priors", "linkinvent.prior"),
            "smiles_file": "input.smi",
        },
    }

    if model_type not in model_configs:
        valid_models = ", ".join(sorted(model_configs.keys()) + ["LibInvent"])
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported values: {valid_models}")

    prefix = REINVENT_PATH
    pool_path = POOL_PATH
    os.makedirs(pool_path, exist_ok=True)

    if not rand_str:
        rand_str = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    job_dir = get_reinvent_job_dir(rand_str)

    json_out_config = os.path.join(job_dir, "_sampling.json")
    output_file = os.path.join(pool_path, f"sampling_{model_type}_{rand_str}.csv")

    selected_config = model_configs[model_type]
    model_file = os.path.join(prefix, selected_config["model_file"])

    parameter_lines = [
        selected_config["description"],
        f'model_file = "{model_file}"',
    ]

    if "smiles_file" in selected_config:
        smiles_file = os.path.join(job_dir, selected_config["smiles_file"])
        parameter_lines.append(f'smiles_file = "{smiles_file}"')

    if "sample_strategy" in selected_config:
        parameter_lines.append(f'sample_strategy = "{selected_config["sample_strategy"]}"')

    if "temperature" in selected_config:
        parameter_lines.append(f'temperature = {selected_config["temperature"]}')

    parameter_lines.extend([
        f"output_file = '{output_file}'  # sampled SMILES and NLL in CSV format",
        f"num_smiles = {num_samples}  # number of SMILES to be sampled, 1 per input SMILES",
        "unique_molecules = true  # if true remove all duplicatesd canonicalize smiles",
        "randomize_smiles = true # if true shuffle atoms in SMILES randomly",
    ])

    seed_line = f"seed = {int(random_seed)}\n" if random_seed is not None else ""
    config_toml = f"""
run_type = "sampling"
device = "{device}"  # set torch device e.g. "cpu"
{seed_line}json_out_config = "{json_out_config}"  # write this TOML to JSON

[parameters]
{os.linesep.join(parameter_lines)}
        """
    
    output_path = os.path.join(job_dir, f"{model_type}.toml")
    with open(output_path, "w") as f:
        f.write(config_toml)

    print(f"Updated TOML file saved to: {output_path}")
    return output_path


def update_libinvent_config(model_type="LibInvent", device='cuda:6', num_samples=200, rand_str=None):
    """
    Create a TOML config file for LibInvent (R-group generation).
    This returns the path to the updated config file.

    Args:
        model_type (str): The model type to use for updating paths. Default is "LibInvent".
    """
    # Fixed input TOML path
    prefix = REINVENT_PATH
    pool_path = POOL_PATH
    # create directories
    os.makedirs(pool_path, exist_ok=True)

    if not rand_str:
        rand_str = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(10))
    job_dir = get_reinvent_job_dir(rand_str)
    json_out_config = os.path.join(job_dir, "_sampling.json")
    model_file = os.path.join(prefix, "priors", "libinvent.prior")
    smiles_file = os.path.join(job_dir, "rgroup.smi")
    output_file = os.path.join(pool_path, f"sampling_REINVENT4_{rand_str}.csv")
    
    config_toml = f"""
run_type = "sampling"
device = "{device}"  # set torch device e.g. "cpu"
json_out_config = "{json_out_config}"  # write this TOML to JSON
[parameters]
## LibInvent: find R-groups for the given scaffolds
model_file = "{model_file}"
smiles_file = "{smiles_file}"  # 1 scaffold per line with attachment points
output_file = '{output_file}'  # sampled SMILES and NLL in CSV format
num_smiles = {num_samples}  # number of SMILES to be sampled, 1 per input SMILES
unique_molecules = true  # if true remove all duplicatesd canonicalize smiles
randomize_smiles = true # if true shuffle atoms in SMILES randomly
        """
    
    output_path = os.path.join(job_dir, f"{model_type}.toml")
    with open(output_path, "w") as f:
        f.write(config_toml)

    print(f"Updated TOML file saved to: {output_path}")
    return output_path


def label_star_atoms(smiles: str) -> str:
    """
    Replace each * or [*] attachment point in a SMILES string with
    sequential labels [*:1], [*:2], and so on.
    """
    index = 1
    # Match either the bracketed [*] form or a standalone *.
    pattern = re.compile(r'\[\*\]|\*')
    def replace(match):
        nonlocal index
        new = f"[*:{index}]"
        index += 1
        return new
    return pattern.sub(replace, smiles)


def save_smi_for_rgroup(scaffold_smiles_list: List[str], rand_str: str=None):
    """
    Save scaffold SMILES to rgroup.smi under the job-specific
    tmp/reinvent/<rand_str> directory, one SMILES string per line.
    
    Args:
        scaffold_smiles_list (List[str]): Scaffold SMILES strings with
            attachment-point labels such as [*:1].
        rand_str (str): Optional identifier for an isolated job directory,
            preventing TOML and SMILES files from different runs from colliding.
    """
    
    # Ensure the job directory exists.
    file_name = os.path.join(get_reinvent_job_dir(rand_str), "rgroup.smi")

    # Label attachment points in every scaffold.
    processed_smiles = []
    for smi in scaffold_smiles_list:
        labeled_smi = label_star_atoms(smi)
        processed_smiles.append(labeled_smi)

    # Write one SMILES string per line, including a final newline.
    with open(file_name, "w") as f:
        f.write("\n".join(processed_smiles) + "\n")
        
    print(f"Total {len(processed_smiles)} scaffold SMILES saved to {file_name}")
    return f"Scaffold smi file updated with {len(processed_smiles)} entries at '{file_name}'"

def run_reinvent(config_file: str):
    """
    Run the REINVENT command with the specified log and configuration files to generate a pool of candidate molecules.
    Args:
        config_file (str): The path to the configuration .toml file.
    """

    log_file = config_file.replace(".toml", ".log")
    # Directly call the Python module to avoid shebang issues
    command = [
        REINVENT_PYTHON, "-m", "reinvent.Reinvent",
        "-l", log_file,
        config_file
    ]
    
    try:
        # Set PYTHONPATH to ensure reinvent package is found
        import os
        env = os.environ.copy()
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            env.pop(key, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
        env['PYTHONPATH'] = REINVENT_PATH + os.pathsep + env.get('PYTHONPATH', '')
        
        # Run the command
        subprocess.run(command, check=True, env=env, cwd=REINVENT_PATH)
        return "REINVENT execution completed successfully."
    except subprocess.CalledProcessError as e:
        return f"Error occurred while running REINVENT: {e}"
    except FileNotFoundError as e:
        return f"Error: Command not found: {e}"
