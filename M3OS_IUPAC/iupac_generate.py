import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1: WARN, 2: ERROR, 3: FATAL

import json
from pathlib import Path

from STOUT import translate_forward


IUPAC_DICT_PATH = Path(__file__).resolve().with_name('smiles_iupac_dict.json')


def get_iupac_names(smiles_list: list[str]) -> dict[str, str | None]:
    '''
    Generate the IUPAC names of multiple molecules from their SMILES representations.

    Args:
        smiles_list (list): List of SMILES representations of molecules.
        
    Returns:
        dict: Dictionary where the key is the SMILES string and the value is the IUPAC name of the molecule,
              or None if an error occurs for any SMILES.
    '''
    result = {}
    cache_updated = False
    
    try:
        with IUPAC_DICT_PATH.open('r', encoding='utf-8') as f:
            smiles_iupac_dict = json.load(f)
    except FileNotFoundError:
        smiles_iupac_dict = {}

    for smiles in smiles_list:
        if smiles in smiles_iupac_dict:
            # If the IUPAC name is already in the dictionary, use it.
            result[smiles] = smiles_iupac_dict[smiles]
        else:
            try:
                iupac_name = translate_forward(smiles)
                smiles_iupac_dict[smiles] = iupac_name
                cache_updated = True
                result[smiles] = iupac_name
            except Exception as e:
                print(f"Error when converting SMILES to IUPAC for {smiles}: {e}")
                result[smiles] = None
    
    # Only write the cache when new entries were added.
    if cache_updated:
        with IUPAC_DICT_PATH.open('w', encoding='utf-8') as f:
            json.dump(smiles_iupac_dict, f, ensure_ascii=False, indent=4)
    
    return result
