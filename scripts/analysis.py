import os
import json
import re

import numpy as np
import pandas as pd


# =====================================
# Interpretation parser
# =====================================

def extract_field(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


# =====================================
# Main parser
# =====================================

def parse_folder(folder_path):

    processed = {}

    for filename in os.listdir(folder_path):

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder_path, filename)

        with open(filepath) as f:
            data = json.load(f)

        processed[filename] = {}

        for seq_name, seq_data in data.items():
            processed[filename][seq_name] = {}

            for interpreter_name, interpretation_data in seq_data.items():

                processed[filename][seq_name][interpreter_name] = {
                    "interpretation": extract_importance_tensor(interpretation_data["interpretation"]),
                    "optimizers": {}
                }

                for optimizer_name, optimizer_data in interpretation_data["optimizers_results"].items():

                    processed[filename][seq_name][interpreter_name]["optimizers"][optimizer_name] = {
                        "best_sequence": optimizer_data["best_sequence"],
                        "trajectory": pd.DataFrame(optimizer_data["trajectory"])
                    }

    return processed

import re
import torch

def extract_importance_tensor(text: str):
    match = re.search(r"importance_scores=tensor\((.*)\),\s*sequence=", text, re.S)
    if not match:
        return None

    tensor_str = match.group(1)

    # zamiana na torch.tensor(...)
    return eval("torch.tensor(" + tensor_str + ")")
