# Heading 1 (Pipeline Command Line Interface Entrypoint)
## Parsing framework execution flags and mapping token sequences onto runner configurations
import argparse
import yaml
from typing import Dict, Any

from PromotorOptimizer.pipeline.configs import PipelineConfig
from PromotorOptimizer.pipeline.runner import PipelineRunner


def parse_args():
    """
    Parses structural execution commands passed via the host terminal interface.

    :return: Parsed arguments namespace mapping pipeline parameter tokens.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Unified sequence optimization and reconstruction high-throughput pipeline."
    )

    # Core file coordinates tracking parameters
    parser.add_argument("--input", required=True, help="Path to the input biological sequence TSV file.")
    parser.add_argument("--output", required=True, help="Path to the destination trajectory save deployment.")

    # Polimorphic loss functions targeting configuration
    parser.add_argument("--objective", required=True, help="Identifier name of the registered target LossObjective class.")
    parser.add_argument("--mutation-budget", type=int, default=None, help="Fallback maximum allowed nucleotide mutations budget.")

    # Architectural execution component lists
    parser.add_argument("--models", nargs="+", required=True, help="List of evaluation model names to resolve from registry.")
    parser.add_argument("--optimizers", nargs="+", required=True, help="List of exploration heuristic optimizers to initialize.")
    parser.add_argument("--interpreters", nargs="+", required=True, help="List of sequence gradient attribution strategies to deploy.")

    # Structural training parameters
    parser.add_argument("--iterations", type=int, default=50, help="Total mutation sequence iterations per sequence trajectory.")
    
    # Extended dictionary configuration injector
    parser.add_argument("--config-path", default=None, help="Path to the experiment YAML file containing deep parameter overrides.")

    return parser.parse_args()


def main():
    """
    Main orchestration function mapping command line tokens directly onto automated runner modules.
    """
    args = parse_args()

    # Load optional yaml config data for nested overrides integration
    ## Initialize empty structures for deep config properties mappings
    objective_config: Dict[str, Any] = {"penalty_std": 0.2}
    validation_config = None
    runtime_overrides: Dict[str, Any] = {}

    if args.config_path:
        with open(args.config_path, "r") as stream:
            yaml_data = yaml.safe_load(stream)
        
        ### Extract raw configurations if zdefiniowane inside the target file
        objective_config = yaml_data.get("objective_config", objective_config)
        validation_config = yaml_data.get("validation_config", None)
        
        ### Structure runtime overrides to match internal wrapper execution payloads
        runtime_overrides = {
            "optimizers": {
                opt_name: {"optimizer_config": opt_cfg}
                for opt_name, opt_cfg in yaml_data.get("optimizer_overrides", {}).items()
            },
            "interpreters": {
                int_name: {"interpreter_config": int_cfg}
                for int_name, int_cfg in yaml_data.get("interpreter_overrides", {}).items()
            }
        }
        
        ### Inject baseline iterations limit within optimizer configs if absent
        for opt_name in runtime_overrides["optimizers"]:
            if "iterations" not in runtime_overrides["optimizers"][opt_name]["optimizer_config"]:
                runtime_overrides["optimizers"][opt_name]["optimizer_config"]["iterations"] = args.iterations

    # Configuration serialization mapping
    ## Bind parsed input flags and dictionaries onto the static PipelineConfig payload
    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        objective=args.objective,
        mutation_budget=args.mutation_budget,
        models=args.models,
        optimizers=args.optimizers,
        interpreters=args.interpreters,
        iterations=args.iterations,
        objective_config=objective_config,
        validation_config=validation_config
    )

    # Execution sweep dispatch
    ## Initialize the runner engine container and execute the graph sequence tracks with overrides
    runner = PipelineRunner(config)
    runner.run(runtime_overrides=runtime_overrides)


if __name__ == "__main__":
    main()