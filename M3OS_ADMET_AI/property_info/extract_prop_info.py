import pandas as pd
import json
from collections import defaultdict
from pathlib import Path


PHYSICOCHEMICAL_PROPERTIES = {
    "molecular_weight": {
        "Description": "Molecular weight",
        "Units": "Dalton",
        "Task_Type": "Regression",
    },
    "logP": {
        "Description": "RDKit Crippen octanol/water partition coefficient (MolLogP)",
        "Units": "log-ratio",
        "Task_Type": "Regression",
    },
    "hydrogen_bond_acceptors": {
        "Description": "Hydrogen bond acceptor count",
        "Units": "count",
        "Task_Type": "Regression",
    },
    "hydrogen_bond_donors": {
        "Description": "Hydrogen bond donor count",
        "Units": "count",
        "Task_Type": "Regression",
    },
    "Lipinski": {
        "Description": "Number of Lipinski rule-of-five criteria satisfied",
        "Units": "count out of 4",
        "Task_Type": "Regression",
    },
    "QED": {
        "Description": "Quantitative estimate of drug-likeness",
        "Units": "unitless (0-1)",
        "Task_Type": "Regression",
    },
    "stereo_centers": {
        "Description": "Stereocenter count",
        "Units": "count",
        "Task_Type": "Regression",
    },
    "tpsa": {
        "Description": "Topological polar surface area",
        "Units": "Å^2",
        "Task_Type": "Regression",
    },
}


def excel_to_custom_json(file_path, output_path):
    # Initialize the nested dictionary.
    result_dict = defaultdict(dict)
    result_dict["Physicochemical"].update(PHYSICOCHEMICAL_PROPERTIES)
    
    # Specify the worksheet names to process.
    sheets = ["TDC Single-Multi Regression", "TDC Single-Multi Classification"]
    
    try:
        # Process each worksheet.
        for sheet_name in sheets:
            # Read the Excel worksheet. Install openpyxl if it is unavailable.
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Process each row.
            for _, row in df.iterrows():
                category = str(row['Category']).strip()
                dataset = str(row['Dataset']).strip()
                description = str(row['Description']).strip()
                
                # Initialize the dataset metadata.
                dataset_info = {
                    "Description": description
                }
                
                # For regression tasks, include the Units column when available.
                if sheet_name == "TDC Single-Multi Regression" and 'Units' in row:
                    # Omit missing units.
                    unit_value = row['Units']
                    if pd.notna(unit_value):
                        dataset_info["Units"] = str(unit_value).strip()

                if sheet_name == "TDC Single-Multi Regression":
                    dataset_info["Task_Type"] = "Regression"
                else:
                    dataset_info["Task_Type"] = "Classification"
                
                # Store metadata under its category and dataset.
                result_dict[category][dataset] = dataset_info

        # Save the metadata as JSON.
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)
            
        print(f"Success: JSON file saved to {output_path}")

    except Exception as e:
        print(f"Error while processing property metadata: {e}")

# Configure input and output paths.
PROPERTY_INFO_DIR = Path(__file__).resolve().parent
file_path = PROPERTY_INFO_DIR / "Supplementary_Table_1.xlsx"
output_json_path = PROPERTY_INFO_DIR / "property_meta_info.json"

if __name__ == "__main__":
    excel_to_custom_json(file_path, output_json_path)
