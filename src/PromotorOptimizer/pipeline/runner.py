# Heading 1 (Pipeline Orchestration Space)
## Core library integrations, framework registries, and data processing interfaces
import json
import os
import pandas as pd
from typing import Dict, Any, Optional, Literal

from ..models.model_manager import ModelManager
from ..models.registry import ModelRegistry
from ..optimizers.depracated.registry import OptimizerRegistry
from ..interpreters.depracated.registry import InterpreterRegistry
from ..core.wrapper import SequencePredictorModelWrapper
from ..utils.logger import get_custom_logger
from .configs import PipelineConfig

# Instantiation Protocol
logger = get_custom_logger(__name__)


class PipelineRunner:
    """
    Unified entrypoint container orchestrating baseline pipeline resources,
    sequence input channels, multi-model evaluation frameworks, and search execution tracks.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initializes the global experiment orchestration architecture components.

        :param config: Strongly typed static property mapping containing baseline pipeline fields.
        :type config: PipelineConfig
        """
        self.config = config
        logger.info("Initializing high-throughput computational pipeline workspace.")

        # Infrastructure initialization
        ## Resolve target computational evaluation networks via ModelRegistry
        logger.info("Loading evaluation models from network repository space: %s", config.models)
        self.models_dict = ModelRegistry.load(config.models)
        self.model_manager = ModelManager(self.models_dict)

        ## Resolve baseline bio-sequence validation parameter configurations
        validation_config = {
            "max_homopolymer_at": 10,
            "max_homopolymer_gc": 7,
            "gc_percent_range": (0.25, 0.65),
            "min_length": 230,
            "max_length": 230
        }
        
        logger.info("Loading heuristic exploration optimizers: %s", config.optimizers)
        self.optimizers = OptimizerRegistry.load(
            config.optimizers,
            validation_config=validation_config
        )

        ## Resolve targeted gradient tracking and attribution frameworks
        logger.info("Loading sequence attribution interpretation interfaces: %s", config.interpreters)
        self.interpreters = InterpreterRegistry.load(config.interpreters)

        # Sequence payload streaming
        ## Load high-throughput target inputs from local system file paths
        logger.info("Reading input biological sequence dataset path: %s", self.config.input_path)
        
        if config.task_mode == "constrained_recovery":
            ### Execute processing track tailored for constrained sequence restoration
            df = pd.read_csv(self.config.input_path, sep="\t", header=0)
            self.sequences = {
                row["id"]: {
                    "sequence": row["sequence"],
                    "mutation_budget": int(row["introduced_mutations"]),
                    "original_activity": float(row["original_activity"])
                }
                for _, row in df.iterrows()
            }
        else:
            ### Execute processing track tailored for unconstrained de novo sequence design
            df = pd.read_csv(self.config.input_path, sep="\t", header=0)
            self.sequences = {
                row["id"]: {
                    "sequence": row["sequence"],
                    "mutation_budget": self.config.mutation_budget,
                    "original_activity": None
                }
                for _, row in df.iterrows()
            }

        logger.debug("Successfully loaded unique target structures volume: %s", len(self.sequences))

        # Structural runtime parameters
        ## Determine active prediction ensemble scaling limits dynamically
        self.model_type: Literal["single", "ensemble"] = (
            "single" if len(config.models) == 1 else "ensemble"
        )

        self.mode: Literal["optimization", "reconstruction"] = (
            "reconstruction"
            if config.task_mode == "constrained_recovery"
            else "optimization"
        )

        # Pipeline wrapper setup
        ## Initialize tracking execution boundaries within the master container wrapper
        logger.info("Building multi-layer execution orchestration wrapper framework.")
        self.wrapper = SequencePredictorModelWrapper(
            model_type=self.model_type,
            mode=self.mode,
            sequences=self.sequences,
            model_manager=self.model_manager,
            optimizers_list=self.optimizers,
            interpreters_list=self.interpreters
        )

        logger.info("Pipeline orchestration architecture instantiated successfully.")

    # -------------------------------------------------
    # MAIN PIPELINE EXECUTION ENTRYPOINT
    # -------------------------------------------------

    def run(self, runtime_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Coordinates the continuous processing loops across sequence targets, passing down
        the optional runtime parameter dictionaries containing configuration blocks.

        :param runtime_overrides: Master configuration block mapping runtime fields (e.g., optimizer_config).
        :type runtime_overrides: dict, optional
        :return: Map structure compiling execution data tracking elements.
        :rtype: dict
        """
        logger.info("Starting pipeline processing loops. Task mode target: %s", self.config.task_mode)
        runtime_overrides = runtime_overrides or {}

        # Mode execution routing
        ## Check active task mode criteria to route underlying tracking data structures
        if self.config.task_mode == "constrained_recovery":
            logger.info("Executing sequence restoration trajectory execution path.")
            
            ### Inject baseline loop parameters if explicit overrides are absent
            if "optimizers" not in runtime_overrides:
                runtime_overrides["optimizers"] = {
                    opt.__class__.__name__: {
                        "optimizer_config": {"iterations": self.config.iterations}
                    }
                    for opt in self.optimizers
                }

            results = self.wrapper.ReconstructSequences(
                override_config=runtime_overrides,
                output_path=self.config.output_path
            )
        else:
            logger.info("Executing sequence design optimization trajectory execution path.")
            
            if "optimizers" not in runtime_overrides:
                runtime_overrides["optimizers"] = {
                    opt.__class__.__name__: {
                        "optimizer_config": {"iterations": self.config.iterations}
                    }
                    for opt in self.optimizers
                }

            results = self.wrapper.OptimizeSequences(
                override_config=runtime_overrides,
                output_path=self.config.output_path
            )

        # Telemetry persistence block
        ## Execute a final storage flush to verify that final data matrices are fully saved to disk
        logger.info("Executing final data synchronization sweep to destination path: %s", self.config.output_path)
        self._final_save_results(results)

        logger.info("High-throughput pipeline task loops completed successfully.")
        return results

    def _final_save_results(self, results: Dict[str, Any]) -> None:
        """
        Saves the final nested dictionary tracking output to disk.

        :param results: Compiled experimental validation trajectory results dictionary.
        :type results: dict
        """
        output_path = self.config.output_path
        output_dir = os.path.dirname(output_path)

        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Encountered unhandled error during final data write sequence.", exc_info=True)