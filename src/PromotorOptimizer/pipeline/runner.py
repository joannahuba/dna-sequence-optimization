# pipeline/runner.py

import json
import os
import pandas as pd
from typing import Literal

from ..models.model_manager import ModelManager
from ..models.registry import ModelRegistry
from ..optimizers.registry import OptimizerRegistry
from ..interpreters.registry import InterpreterRegistry

from ..core.wrapper import SequencePredictorModelWrapper
from .configs import PipelineConfig


class PipelineRunner:
    """
    Single entrypoint for all experiments:
    - constrained recovery
    - unconstrained design
    """

    def __init__(self, config: PipelineConfig):

        self.config = config

        print("\n[INFO] Initializing pipeline")

        # 1. LOAD MODELS
        print(f"[INFO] Loading models: {config.models}")
        self.models_dict = ModelRegistry.load(config.models)
        self.model_manager = ModelManager(self.models_dict)

        # 2. LOAD OPTIMIZERS
        print(f"[INFO] Loading optimizers: {config.optimizers}")
        validation_config = {
            "max_homopolymer_at": 10,
            "max_homopolymer_gc": 7,
            "gc_percent_range": (0.25, 0.65),
            "min_length": 230,
            "max_length": 230
        }
        self.optimizers = OptimizerRegistry.load(
            config.optimizers,
            validation_config=validation_config
        )

        # 3. LOAD INTERPRETERS
        print(f"[INFO] Loading interpreters: {config.interpreters}")
        self.interpreters = InterpreterRegistry.load(config.interpreters)

        # 4. LOAD INPUT SEQUENCES
        print(f"[INFO] Reading input file: {self.config.input_path}")

        # 4. LOAD INPUT SEQUENCES
        print(
            f"[INFO] Reading input file: "
            f"{self.config.input_path}"
        )

        if config.task_mode == "constrained_recovery":

            df = pd.read_csv(
                self.config.input_path,
                sep="\t",
                header=0
            )

            self.sequences = {
                row["id"]: {
                    "sequence":
                        row["sequence"],

                    "mutation_budget":
                        int(
                            row[
                                "introduced_mutations"
                            ]
                        ),

                    "original_activity":
                        float(
                            row[
                                "original_activity"
                            ]
                        )
                }
                for _, row in df.iterrows()
            }

        else:

            df = pd.read_csv(
                self.config.input_path,
                # DONE : change format 
                sep="\t",
                header=0,
            )

            self.sequences = {
                row["id"]: {
                    "sequence":
                        row["sequence"],

                    "mutation_budget":
                        self.config.mutation_budget,

                    "original_activity":
                        None
                }
                for _, row in df.iterrows()
            }

        print(df)

        print(
            f"[INFO] Loaded "
            f"{len(self.sequences)} sequences"
        )

        print(f"[INFO] Loaded {len(self.sequences)} sequences")

        # 5. MODEL TYPE
        self.model_type: Literal["single", "ensemble"] = (
            "single" if len(config.models) == 1 else "ensemble"
        )

        # 6. MODE
        self.mode: Literal["optimization", "reconstruction"] = (
            "reconstruction"
            if config.task_mode == "constrained_recovery"
            else "optimization"
        )

        # 7. WRAPPER
        print("[INFO] Building wrapper")


        self.wrapper = SequencePredictorModelWrapper(
            model_type=self.model_type,
            mode=self.mode,
            sequences=self.sequences,
            model_manager=self.model_manager,
            optimizers_list=self.optimizers,
            interpreters_list=self.interpreters
        )

        print("[INFO] Pipeline initialized successfully")

    # -------------------------
    # MAIN ENTRYPOINT
    # -------------------------
    def run(self):

        print(f"\n[INFO] Task mode: {self.config.task_mode}")

        if self.config.task_mode == "constrained_recovery":

            print("[INFO] Running reconstruction pipeline")
            
            # TODO mutation_budget, org_expression aby to jakoś wczytywać
            results = self.wrapper.ReconstructSequences(
                reconstruction_config={
                    "iterations": self.config.iterations
                }
            )

        else:

            print("[INFO] Running optimization pipeline")

            results = self.wrapper.OptimizeSequences(
                config={
                    "iterations": self.config.iterations
                }
            )

        # -------------------------
        # SAVE RESULTS
        # -------------------------
        self._save_results(results)

        print("[INFO] Pipeline finished successfully")

        return results

    # -------------------------
    # OUTPUT HANDLER
    # -------------------------
    def _save_results(self, results):

        output_path = self.config.output_path

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"[INFO] Saving results to: {output_path}")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)