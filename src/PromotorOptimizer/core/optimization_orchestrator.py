# Heading 1 (Trajectory Orchestration Space)
## Module infrastructure imports, typing boundaries, tensor backends, and serialization tools
import copy
from typing import Dict, List, Any, Tuple
import numpy as np
import torch

from ..models.model_manager import ModelManager
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
        interpreter: Any
    ):
        """
        Initializes the dynamic trajectory execution engine.

        :param model_manager: Global computational container executing network inferences.
        :type model_manager: ModelManager
        :param optimizer: Reconfigured stateful search heuristic engine.
        :type optimizer: Any
        :param interpreter: Decoupled functional matrix attribution strategy.
        :type interpreter: Any
        """
        self.model_manager = model_manager
        self.optimizer = optimizer
        self.interpreter = interpreter

    def _initialize_tracking_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Resolves targeted tracking dictionary nodes based on the interpreter schema flags.

        :param registered_models: List of available model identifiers inside the manager instance.
        :type registered_models: List[str]
        :return: Normalized layout configuration mapping tracking nodes dynamically.
        :rtype: Dict[str, Any]
        """
        logger.debug("Resolving strategy layer metadata schema outputs.")
        schema_meta = self.interpreter.resolve_output_schema(registered_models)
        models_tracking_payload = {}
        
        # Schema distribution logic
        ## Differentiate tracking structures for aggregate ensembles vs isolated models
        if schema_meta.get("is_aggregated", False):
            ### Execute Option B pathing: assign a unified key with explicit component lists
            models_tracking_payload["aggregated_models"] = {
                "model_config": {},
                "evaluated_models": registered_models,
                "steps": []
            }
        else:
            ### Execute discrete pathing: map each model architecture to its own top-level profile key
            for model_name in registered_models:
                models_tracking_payload[model_name] = {
                    "model_config": {},
                    "evaluated_models": [model_name],
                    "steps": []
                }
                
        return models_tracking_payload

    def run_trajectory(
        self,
        initial_sequence: str,
        runtime_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Drives the iterative optimization timeline for a single sequence,
        converting text arrays into tensor representations for batch processing.

        :param initial_sequence: Starting wild-type or seed nucleotide sequence string.
        :type initial_sequence: str
        :param runtime_context: Unified configuration payload passed from the orchestration layer.
        :type runtime_context: Dict[str, Any]
        :return: Structured nested map compiling execution metrics matching the JSON specification.
        :rtype: Dict[str, Any]
        """
        logger.info("Launching isolated execution track for target sequence profile.")
        optimizer_config = runtime_context.get("optimizer_config", {})
        iterations = optimizer_config.get("iterations", 50)
        
        registered_models = self.model_manager.get_model_names()
        models_tracking_payload = self._initialize_tracking_schema(registered_models)
        
        # State instantiation phase
        ## Initialize persistence dictionaries tracking the exploration search state
        search_state = self.optimizer.initialize_search_state(initial_sequence, optimizer_config)
        current_sequence = initial_sequence

        # Time step integration loop
        ## Drive continuous optimization trajectories across the defined iterations timeline
        for it in range(iterations):
            logger.info("Executing trajectory step evaluation cycle: %s / %s", it + 1, iterations)
            
            # Batch evaluation setup
            ## Compute unified raw sequence predictions for all models to prevent redundant processing overhead
            prediction_map = self.model_manager.predict_sequences([current_sequence])
            
            ## Resolve active tensor computational device execution targets from the model manager
            device = self.model_manager.get_device()
            X_encoded = encode_batch([current_sequence])
            tensor_x = torch.tensor(X_encoded, dtype=torch.float32, device=device)
            
            global_importance_maps = {}
            
            # Schema tracking loop
            ## Iterate over configured target key blocks defined by the architectural payload schema
            # TODO it is not optimal w can write it in to separate loops (one for aggregated and on for separate) 
            # ładowaś wszystkich na raz się boje ... .
            for target_key, target_meta in models_tracking_payload.items():
                models_predictions = {}
                models_attributions = {}
                
                ## Process individual evaluated models assigned to the current tracking target
                for model_name in target_meta["evaluated_models"]:
                    meta = self.model_manager.get_models()[model_name]
                    model_instance = meta["model"]
                    
                    ### Synchronize target model architecture with the designated active compute device
                    model_instance.to(device)
                    
                    ### Extract position attribution scores via the strategy execution layer
                    attribution_matrix = self.interpreter.compute_tensor_attribution(tensor_x, model_instance)
                    
                    models_predictions[model_name] = prediction_map[current_sequence].get(model_name, 0.0)
                    models_attributions[model_name] = attribution_matrix
                    global_importance_maps[model_name] = attribution_matrix
                    
                ### Compile execution step records directly into the active trajectory tracking array
                step_record = {
                    "iteration": it,
                    "current_sequence": current_sequence,
                    "models_predictions": models_predictions,
                    "models_attributions": models_attributions
                }
                target_meta["steps"].append(step_record)
            
            # Sequence variation tracking
            ## Suggest validation-filtered mutant candidate lists based on computed weights
            candidate_pool = self.optimizer.generate_candidate_pool(search_state, global_importance_maps)
            
            if not candidate_pool:
                logger.info("Candidate generator space returned empty pool. Terminating search early.")
                break
                
            ## Evaluate fitness rankings for candidate variants in a unified batch operation
            scored_candidates = []
            fitness_evaluations = self.model_manager.score_sequences(candidate_pool)
            for candidate in candidate_pool:
                fitness_score = fitness_evaluations[candidate]["fitness"]
                scored_candidates.append((candidate, fitness_score))
                
            ## Apply selection filters to transition the internal search parameters
            search_state = self.optimizer.update_generation_step(search_state, scored_candidates)
            current_sequence = search_state.get("best_sequence", current_sequence)

        return {
            "models": models_tracking_payload
        }