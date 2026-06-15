# Heading 1 (Core Pipeline Orchestration Space)
## Module infrastructure imports, trajectory routers, and serialization utilities
import json
import os
import time
from typing import Dict, List, Optional, Any, Literal

from .optimization_orchestrator import TrajectoryOrchestrator
from ..models.model_manager import ModelManager
from ..loss_functions import BaseLossObjective
from ..utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class SequencePredictorModelWrapper:
    """
    Orchestration wrapper managing execution flows across multi-target sequences,
    optimizers, and structural model importance interpreters using universal objective configurations.
    """

    def __init__(
        self,
        model_type: Literal["ensemblae", "single"],
        sequences: Dict[str, Dict[str, Any]],
        model_manager: ModelManager,
        optimizers_list: List[Any],
        interpreters_list: List[Any],
        objective: BaseLossObjective,
        prefix_len: int = 0,
        suffix_len: int = 0
    ):
        """
        Initializes the sequence execution tracking wrapper.

        :param model_type: Flag specifying single or ensemble network evaluations.
        :type model_type: str
        :param sequences: Nested mapping containing targets and experimental boundaries.
        :type sequences: dict
        :param model_manager: Global computational execution container for target networks.
        :type model_manager: ModelManager
        :param optimizers_list: Instantiated sequence search heuristics.
        :type optimizers_list: list
        :param interpreters_list: Gradient tracker and molecular attribution frameworks.
        :type interpreters_list: list
        :param objective: Injected evaluation criteria driving fitness and loss tracking paths.
        :type objective: BaseLossObjective
        :param prefix_len: Number of unmutable flanking nucleotides at the sequence start.
        :type prefix_len: int
        :param suffix_len: Number of unmutable flanking nucleotides at the sequence end.
        :type suffix_len: int
        """
        self.model_type = model_type
        self.sequences = sequences
        self.model_manager = model_manager
        self.optimizers_list = optimizers_list
        self.interpreters_list = interpreters_list
        self.objective = objective
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.output = {}

    def _incremental_checkpoint(self, output_path: Optional[str]) -> None:
        """
        Serializes current trajectory states directly to file storage targets.

        :param output_path: Destination filesystem string path.
        :type output_path: str, optional
        """
        if not output_path:
            return
        try:
            dir_name = os.path.dirname(output_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(output_path, "w") as out_file:
                json.dump(self.output, out_file, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to execute incremental pipeline checkpoint save.", exc_info=True)

    # -------------------------------------------------
    # UNIFIED TRAJECTORY EXECUTION ENGINE
    # -------------------------------------------------

    def ExecuteTrajectories(
        self,
        override_config: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes structural sequence transformation trajectories across all registered targets 
        using a polymorphic evaluation approach driven by the injected objective layer.

        :param override_config: Runtime dictionary containing external parameter overrides.
        :type override_config: dict, optional
        :param output_path: Local tracking output path for executing file checkpoints.
        :type output_path: str, optional
        :return: Map structure compiling tracking information across internal steps.
        :rtype: dict
        """
        logger.info("Initializing high-throughput unified trajectory optimization loops.")
        override_config = override_config or {}
        total_sequences = len(self.sequences)
        registered_models = self.model_manager.get_model_names()

        # Processing loop
        ## Process target sequences sequentially
        for idx, (seq_id, seq_data) in enumerate(self.sequences.items(), start=1):
            start_time = time.time()
            seq = seq_data["sequence"]
            mutation_budget = seq_data.get("mutation_budget", None)
            target_value = seq_data.get("target_value", None)

            logger.info("Processing trajectory sequence %s/%s | ID: %s", idx, total_sequences, seq_id)

            ## Clean biological tracking sequence metadata profile allocation
            self.output[seq_id] = {
                "sequence_metadata": {
                    "initial_sequence": seq,
                    "mutation_budget": mutation_budget,
                    "target_value": target_value
                },
                "optimizers": {}
            }

            ## Execute registered search heuristics sequentially
            for optimizer in self.optimizers_list:
                optimizer_name = optimizer.__class__.__name__
                optimizer_payload = override_config.get("optimizers", {}).get(optimizer_name, {})
                optimizer_config = optimizer_payload.get("optimizer_config", {})

                self.output[seq_id]["optimizers"][optimizer_name] = {
                    "optimizer_config": optimizer_config,
                    "interpreters": {}
                }

                ## Execute active molecular attribution interpreters inside the optimizer path
                for interpreter in self.interpreters_list:
                    interpreter_name = interpreter.__class__.__name__
                    interpreter_payload = override_config.get("interpreters", {}).get(interpreter_name, {})
                    interpreter_config = interpreter_payload.get("interpreter_config", {})

                    # Interpreter configuration encapsulation
                    ## Embed cost function meta definitions inside the interpreter context where it belongs
                    interpreter_config["objective_type"] = self.objective.__class__.__name__
                    interpreter_config["objective_config"] = getattr(self.objective, "config", {})

                    ### Inject structural boundary controls and target value limits into context
                    runtime_context = {
                        "model_type": self.model_type,
                        "target_value": target_value,
                        "mutation_budget": mutation_budget,
                        "optimizer_config": optimizer_config,
                        "interpreter_config": interpreter_config,
                        "prefix_len": self.prefix_len,
                        "suffix_len": self.suffix_len
                    }

                    orchestrator = TrajectoryOrchestrator(
                        model_manager=self.model_manager,
                        optimizer=optimizer,
                        interpreter=interpreter,
                        objective=self.objective
                    )

                    ### Inspect output schema to determine structural path distribution
                    schema_meta = interpreter.resolve_output_schema(registered_models)
                    
                    self.output[seq_id]["optimizers"][optimizer_name]["interpreters"][interpreter_name] = {
                        "interpreter_config": interpreter_config,
                        "models": {}
                    }

                    if schema_meta.get("is_aggregated", False):
                        #### Execute unified ensemble execution path
                        trajectory_results = orchestrator.run_trajectory(
                            initial_sequence=seq,
                            runtime_context=runtime_context,
                            target_model=None
                        )
                        self.output[seq_id]["optimizers"][optimizer_name]["interpreters"][interpreter_name]["models"] = trajectory_results.get("models", {})
                        self._incremental_checkpoint(output_path)
                    else:
                        #### Execute discrete model execution tracks independently over structural keys
                        for model_name in registered_models:
                            logger.info("Launching isolated model track optimization loop for model: %s", model_name)
                            trajectory_results = orchestrator.run_trajectory(
                                initial_sequence=seq,
                                runtime_context=runtime_context,
                                target_model=model_name
                            )
                            ### Append the model standalone history node directly into the primary tracking map
                            self.output[seq_id]["optimizers"][optimizer_name]["interpreters"][interpreter_name]["models"][model_name] = trajectory_results["models"][model_name]
                            
                            #### Flush checkpoint records incrementally immediately after each individual model trajectory ends
                            self._incremental_checkpoint(output_path)

            elapsed = time.time() - start_time
            logger.info("Completed tracking operations for sequence %s in %2.f seconds.", seq_id, elapsed)

        return self.output