# Heading 1 (Pipeline Orchestration Space)
## Core library integrations, framework registries, and data processing interfaces
import json
import os
import pandas as pd
from typing import Dict, Any, Optional, List

from ..models.model_manager import ModelManager
from ..models.registry import ModelRegistry
from ..optimizers.registry import OptimizerRegistry
from ..interpreters.registry import InterpreterRegistry
from ..loss_functions.registry import ObjectiveRegistry
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

        # Polimorphic Objective Instantiation
        ## Load the universal objective function directly via the central ObjectiveRegistry
        logger.info("Loading universal loss objective strategy: %s", config.objective)
        self.objective = ObjectiveRegistry.load(
            name=config.objective,
            objective_config=config.objective_config
        )

        # Infrastructure initialization
        ## Resolve target computational evaluation networks via ModelRegistry
        logger.info("Loading evaluation models from network repository space: %s", config.models)
        self.models_dict = ModelRegistry.load(config.models)
        self.model_manager = ModelManager(self.models_dict)

        ## Resolve baseline bio-sequence validation parameter configurations
        validation_config = config.validation_config or {
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

        ## Resolve targeted gradient tracking and attribution frameworks, injecting the loaded objective
        logger.info("Loading sequence attribution interpretation interfaces: %s", config.interpreters)
        self.interpreters = InterpreterRegistry.load(
            names=config.interpreters,
            objective=self.objective
        )

        # Sequence payload streaming
        ## Load high-throughput target inputs from local system file paths with positional tolerance
        logger.info("Reading input biological sequence dataset path: %s", self.config.input_path)
        df = pd.read_csv(self.config.input_path, sep="\t", header=None)

        self.sequences = {}
        
        ## Process individual lines dynamically to extract sequence-specific parameters into metadata
        for _, row in df.iterrows():
            ### Map positional indices directly to protect against missing header structures
            seq_id = str(row.iloc[0])
            sequence_str = str(row.iloc[1])
            
            ### Safely cast sequence-specific mutation budgets if present in the data row
            mutation_budget = int(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else config.mutation_budget
            
            ### Safely cast sequence-specific target baseline scores (e.g., original activity) if provided
            target_value = float(row.iloc[3]) if len(row) > 3 and pd.notna(row.iloc[3]) else None

            self.sequences[seq_id] = {
                "sequence": sequence_str,
                "mutation_budget": mutation_budget,
                "target_value": target_value
            }

        logger.debug("Successfully loaded unique target structures volume: %s", len(self.sequences))

        # Structural runtime parameters
        ## Determine active prediction ensemble scaling limits dynamically
        self.model_type = "single" if len(config.models) == 1 else "ensemble"

        # Pipeline wrapper setup
        ## Initialize tracking execution boundaries within the master container wrapper
        logger.info("Building multi-layer execution orchestration wrapper framework.")
        self.wrapper = SequencePredictorModelWrapper(
            model_type=self.model_type,
            sequences=self.sequences,
            model_manager=self.model_manager,
            optimizers_list=self.optimizers,
            interpreters_list=self.interpreters,
            objective=self.objective
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
        logger.info("Starting unified pipeline trajectory execution loops.")
        runtime_overrides = runtime_overrides or {}

        ## Inject baseline loop parameters if explicit overrides are absent
        if "optimizers" not in runtime_overrides:
            runtime_overrides["optimizers"] = {
                opt.__class__.__name__: {
                    "optimizer_config": {"iterations": self.config.iterations}
                }
                for opt in self.optimizers
            }

        # Polimorphic execution routing
        ## Execute the single consolidated trajectory loop driven entirely by the embedded loss objective
        results = self.wrapper.ExecuteTrajectories(
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