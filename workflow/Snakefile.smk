# Heading 1 (Master Pipeline Orchestrator)
## Global graph definition, rules inclusions, and target evaluations execution entries
import os
import glob

# Include internal rules dependencies
include: "rules/common.smk"
include: "rules/run_optimizer.smk"

# Target resolution phase
## Discover all experiment configuration definitions in the repository workspace
EXPERIMENT_FILES = glob.glob("config/experiments/*.yaml")
EXPERIMENTS = [os.path.splitext(os.path.basename(f))[0] for f in EXPERIMENT_FILES]

def get_all_wanted_outputs():
    """
    Computes the complete list of target outputs based on experiment-to-file mappings.
    """
    output_targets = []
    for exp in EXPERIMENTS:
        ### Resolve filenames dynamically per experiment configuration
        filenames = resolve_experiment_inputs(exp)
        for fname in filenames:
            base_name = os.path.splitext(fname)[0]
            target_path = os.path.join(
                GLOBAL_CONFIG["paths"]["results_directory"], 
                exp, 
                f"{base_name}_trajectory.json"
            )
            output_targets.append(target_path)
    return output_targets


rule all:
    input:
        get_all_wanted_outputs()