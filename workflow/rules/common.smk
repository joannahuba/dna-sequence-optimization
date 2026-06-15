# Heading 1 (Snakemake Common Infrastructure Module)
## Module configurations parsing, path resolutions, and dynamic input file routers
import os
import glob
import yaml

# Configuration initialization
## Load global environmental settings from root configuration path
configfile: "config/global_config.yaml"

GLOBAL_CONFIG = config
EXP_DIR = "config/experiments"
INPUT_BASE_DIR = GLOBAL_CONFIG["paths"]["input_directory"]

def get_experiment_config(experiment_name: str) -> dict:
    """
    Reads and parses a specific experiment YAML configuration file.
    """
    config_path = os.path.join(EXP_DIR, f"{experiment_name}.yaml")
    if not os.path.exists(config_path):
        raise ValueError(f"Requested experiment configuration file not found: {config_path}")
    with open(config_path, "r") as stream:
        return yaml.safe_load(stream)

def resolve_experiment_inputs(experiment_name: str) -> list:
    """
    Resolves input files for a given experiment, falling back to all directory contents if null.
    """
    exp_cfg = get_experiment_config(experiment_name)
    explicit_files = exp_cfg.get("input_files")

    if explicit_files is not None:
        ### Return explicit files mapped to their expected base directory
        return [os.path.basename(f) for f in explicit_files]
    
    ### Dynamic fallback path: scan input directory for all matching biological data assets
    search_pattern = os.path.join(INPUT_BASE_DIR, "*.tsv")
    found_files = glob.glob(search_pattern)
    
    return [os.path.basename(f) for f in found_files]