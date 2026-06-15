# Heading 1 (Trajectory Orchestration Space)
## Module infrastructure imports, typing boundaries, tensor backends, and serialization tools
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

from ..models.model_manager import ModelManager
from ..loss_functions import BaseLossObjective
from ..utils.preprocessing import encode_batch
from ..utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class TrajectoryOrchestrator:
    """
    Executes and tracks continuous optimization timelines for individual target sequences
    by managing interactions between reactive search components and attribution strategies.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        optimizer: Any,
        interpreter: Any,
        objective: BaseLossObjective
    ):
        """
        Initializes the dynamic trajectory execution engine.

        :param model_manager: Global computational container executing network inferences.
        :type model_manager: ModelManager
        :param optimizer: Reconfigured stateful search heuristic engine.
        :type optimizer: Any
        :param interpreter: Decoupled functional matrix attribution strategy.
        :type interpreter: Any
        :param objective: Injected loss function contract decoupling state evaluation logic.
        :type objective: BaseLossObjective
        """
        self.model_manager = model_manager
        self.optimizer = optimizer
        self.interpreter = interpreter
        self.objective = objective

    def run_trajectory(
        self,
        initial_sequence: str,
        runtime_context: Dict[str, Any],
        target_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Routes execution paths to specialized trajectory sub-methods based on aggregation schemas.

        :param initial_sequence: Starting wild-type or seed nucleotide sequence string.
        :type initial_sequence: str
        :param runtime_context: Unified configuration payload passed from the orchestration layer.
        :type runtime_context: Dict[str, Any]
        :param target_model: Explicit identifier specifying an isolated model execution target.
        :type target_model: Optional[str]
        :return: Structured nested map compiling execution metrics matching the JSON specification.
        :rtype: Dict[str, Any]
        """
        if target_model is not None:
            return self._run_discrete_trajectory(initial_sequence, runtime_context, target_model)
        return self._run_aggregated_trajectory(initial_sequence, runtime_context)

    def _run_aggregated_trajectory(
        self,
        initial_sequence: str,
        runtime_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Drives iterative multi-model ensemble trajectories with persistent memory residents.

        :param initial_sequence: Starting wild-type or seed nucleotide sequence string.
        :type initial_sequence: str
        :param runtime_context: Unified configuration payload passed from the orchestration layer.
        :type runtime_context: Dict[str, Any]
        :return: Aggregated tracking configuration layout containing step traces.
        :rtype: Dict[str, Any]
        """
        logger.info("Initiating ensemble execution track.")
        optimizer_config = runtime_context.get("optimizer_config", {})
        iterations = optimizer_config.get("iterations", 50)
        device = self.model_manager.get_device()
        registered_models = self.model_manager.get_model_names()

        # Schema structural allocation
        models_tracking_payload = {
            "aggregated_models": {
                "model_config": {},
                "evaluated_models": registered_models,
                "steps": []
            }
        }

        # High-efficiency pre-loading phase
        ## Push all registered models to GPU memory once to eliminate pipeline shuffling bottlenecks
        for model_name in registered_models:
            meta = self.model_manager.get_models()[model_name]
            meta["model"].to(device)
            meta["model"].eval()

        search_state = self.optimizer.initialize_search_state(initial_sequence, optimizer_config)
        current_sequence = initial_sequence

        # Evolutionary optimization timeline loop
        for it in range(iterations):
            logger.info("Aggregated trajectory execution cycle: %s / %s", it + 1, iterations)
            
            prediction_map = self.model_manager.predict_sequences([current_sequence])
            X_encoded = encode_batch([current_sequence])
            tensor_x = torch.tensor(X_encoded, dtype=torch.float32, device=device)
            
            global_importance_maps = {}
            models_predictions = {}
            models_attributions = {}

            ## Parallel batch execution over model weights residing on device cache
            for model_name in registered_models:
                model_instance = self.model_manager.get_models()[model_name]["model"]
                attribution_matrix = self.interpreter.compute_tensor_attribution(tensor_x, model_instance, runtime_context)
                
                models_predictions[model_name] = prediction_map[current_sequence].get(model_name, 0.0)
                models_attributions[model_name] = attribution_matrix
                global_importance_maps[model_name] = attribution_matrix

            # Extended state logging block
            ## Capture heuristic population matrices and weight configurations directly from search_state
            step_record = {
                "iteration": it,
                "current_sequence": current_sequence,
                "models_predictions": models_predictions,
                "models_attributions": models_attributions,
                "beam_population": search_state.get("beam_population", []),
                "weights": search_state.get("weights", {})
            }
            models_tracking_payload["aggregated_models"]["steps"].append(step_record)

            candidate_pool = self.optimizer.generate_candidate_pool(search_state, global_importance_maps)
            if not candidate_pool:
                logger.info("Candidate generator returned empty pool. Halting execution path.")
                break

            scored_candidates = []
            raw_predictions = self.model_manager.predict_sequences(candidate_pool)

            ## Evaluate fitness using polymorphically isolated evaluation metrics
            for candidate in candidate_pool:
                scores = raw_predictions[candidate]
                
                ### Delegate fitness ranking to the injected objective function contract
                fitness_score = self.objective.evaluate_numpy_fitness(
                    predictions=scores,
                    sequence=candidate,
                    metadata=runtime_context
                )
                scored_candidates.append((candidate, fitness_score))

            search_state = self.optimizer.update_generation_step(search_state, scored_candidates)
            current_sequence = search_state.get("best_sequence", current_sequence)

            # Invariant Memory Cleanup Block
            ## Release reference blocks and flush PyTorch autograd allocation caches to prevent OOM anomalies
            del tensor_x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"models": models_tracking_payload}

    def _run_discrete_trajectory(
        self,
        initial_sequence: str,
        runtime_context: Dict[str, Any],
        target_model: str
    ) -> Dict[str, Any]:
        """
        Drives iterative standalone trajectories optimized for an isolated network model.

        :param initial_sequence: Starting wild-type or seed nucleotide sequence string.
        :type initial_sequence: str
        :param runtime_context: Unified configuration payload passed from the orchestration layer.
        :type runtime_context: Dict[str, Any]
        :param target_model: Explicit identifier of the single model targeted for sequence mutations.
        :type target_model: str
        :return: Discrete tracking configuration layout containing step traces for the target key.
        :rtype: Dict[str, Any]
        """
        logger.info("Initiating discrete model isolated execution track.")
        optimizer_config = runtime_context.get("optimizer_config", {})
        iterations = optimizer_config.get("iterations", 50)
        device = self.model_manager.get_device()

        # Schema structural allocation for single key footprint optimization
        models_tracking_payload = {
            target_model: {
                "model_config": {},
                "evaluated_models": [target_model],
                "steps": []
            }
        }

        # Single-model memory retention setup
        ## Lock the single target architecture into GPU memory once before starting iteration tracks
        meta = self.model_manager.get_models()[target_model]
        model_instance = meta["model"]
        model_instance.to(device)
        model_instance.eval()

        search_state = self.optimizer.initialize_search_state(initial_sequence, optimizer_config)
        current_sequence = initial_sequence

        # Evolutionary optimization timeline loop
        for it in range(iterations):
            logger.info("Discrete trajectory execution cycle for model %s: %s / %s", target_model, it + 1, iterations)
            
            # Direct single-model optimization tracking pass
            prediction_map = self.model_manager.predict_sequences([current_sequence], model_names=target_model)
            X_encoded = encode_batch([current_sequence])
            tensor_x = torch.tensor(X_encoded, dtype=torch.float32, device=device)
            
            attribution_matrix = self.interpreter.compute_tensor_attribution(tensor_x, model_instance, runtime_context)
            
            # Package attribution metrics under target model name to match optimizer interface signatures
            global_importance_maps = {target_model: attribution_matrix}
            
            models_predictions = {target_model: prediction_map[current_sequence].get(target_model, 0.0)}
            models_attributions = {target_model: attribution_matrix}

            # Extended state logging block
            ## Capture heuristic population matrices and weight configurations directly from search_state
            step_record = {
                "iteration": it,
                "current_sequence": current_sequence,
                "models_predictions": models_predictions,
                "models_attributions": models_attributions,
                "beam_population": search_state.get("beam_population", []),
                "weights": search_state.get("weights", {})
            }
            models_tracking_payload[target_model]["steps"].append(step_record)

            candidate_pool = self.optimizer.generate_candidate_pool(search_state, global_importance_maps)
            if not candidate_pool:
                logger.info("Candidate generator returned empty pool. Halting discrete track.")
                break

            scored_candidates = []
            # Compute fast batch predictions targeted strictly to our model profile
            raw_predictions = self.model_manager.predict_sequences(candidate_pool, model_names=target_model)

            ## Evaluate fitness using polymorphically isolated evaluation metrics
            for candidate in candidate_pool:
                scores = raw_predictions[candidate]
                
                ### Delegate fitness ranking to the injected objective function contract
                fitness_score = self.objective.evaluate_numpy_fitness(
                    predictions=scores,
                    sequence=candidate,
                    metadata=runtime_context
                )
                scored_candidates.append((candidate, fitness_score))

            search_state = self.optimizer.update_generation_step(search_state, scored_candidates)
            current_sequence = search_state.get("best_sequence", current_sequence)

            # Invariant Memory Cleanup Block
            ## Release reference blocks and flush PyTorch autograd allocation caches to prevent OOM anomalies
            del tensor_x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"models": models_tracking_payload}