# optimizers/simulated_annealing.py

import math
import random
import numpy as np
from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .validator import SequenceValidator


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    An algorithmic sequence optimizer implementing discrete Simulated Annealing.

    It escapes local local sub-optimal minima by allowing transient stochastically 
    bounded drops in regulatory objective scores based on a Metropolis acceptance criterion.

    :param validation_config: Configuration keys mapping tracking constraints for validation checks.
    :type validation_config: dict
    :param initial_temperature: The starting scalar control state for the cooling map. Default is 10.0.
    :type initial_temperature: float
    :param cooling_rate: Geometric degradation multiplier applied per iteration loop step. Default is 0.95.
    :type cooling_rate: float
    """

    def __init__(
        self,
        validation_config,
        initial_temperature=10.0,
        cooling_rate=0.95
    ):
        # Resource initialization
        ## Bind short-circuit structural validator
        self.validator = SequenceValidator(validation_config)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def optimize(
        self,
        sequence,
        model_manager,
        interpreter,
        config
    ):
        """
        Executes a stochastically guided cooling track to optimize the DNA input string.

        This method dynamically re-computes importance attribution scores for the 
        active sequence coordinate state during each iteration to handle epistatic regulatory changes.

        :param sequence: Target wild-type or broken base string context.
        :type sequence: str
        :param model_manager: Unified ensemble model coordination stack interface.
        :type model_manager: ModelManager
        :param interpreter: Live implementation instance of BaseInterpreter to re-evaluate gradients.
        :type interpreter: BaseInterpreter
        :param config: Runtime dictionary payload mapping target modes, labels, and mutation limits.
        :type config: dict
        :return: Map structure compiling the optimal string sequence and performance log tracks.
        :rtype: dict
        """
        # Parse runtime parameters
        ## Extract configuration modes and bounds
        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)
        iterations = config.get("iterations", 100)
        model_type = config.get("model_type", "ensemble")

        prefix_len = 0
        suffix_len = 0

        for model_meta in model_manager.get_models().values():
            dataset_class = model_meta.get("dataset_class")
            if dataset_class and dataset_class.__name__ == "DNADatasetNoAdapters":
                try:
                    input_file = config.get("input_path", "data/reconstruction_input.tsv")
                    temp_dataset = dataset_class(input_file)
                    prefix_len = getattr(temp_dataset, "prefix_len", 0)
                    suffix_len = getattr(temp_dataset, "suffix_len", 0)
                    break
                except Exception as e:
                    pass

        # Define localized performance evaluators
        ## Raw prediction averaging
        def score(seq):
            result = model_manager.predict_sequences([seq])
            return sum(result[seq].values()) / len(result[seq])

        ## Absolute objective distance evaluation
        def reconstruction_score(seq):
            return -abs(score(seq) - target_expression)

        # Optimization track setup
        ## Initialize baseline states
        current_seq = sequence
        current_score = reconstruction_score(sequence) if method == "reconstruction" else score(sequence)
        
        best_seq = current_seq
        best_score = current_score
        
        temperature = self.initial_temperature
        trajectory = []

        # Core optimization loop execution
        for it in range(iterations):
            ## Compute live position sensitivity profiles dynamically for the current sequence state
            interpretation = interpreter.explain(
                model_manager=model_manager,
                sequence=current_seq,
                model_type=model_type
            )
            importance = interpretation.importance_scores

            ## Generate candidate via directed mutation mutations
            ### Draw mutation step single position parameters
            candidate_seq = MutationGenerator.hybrid_mutation(
                sequence=current_seq,
                importance_scores=importance,
                n_mutations=1,
                prefix_len=prefix_len,
                suffix_len=suffix_len
            )

            ## Validate structural and biological properties
            if not self.validator.is_valid(candidate_seq):
                continue

            ## Score candidate variation
            candidate_score = (
                reconstruction_score(candidate_seq)
                if method == "reconstruction"
                else score(candidate_seq)
            )

            ## Evaluate state transition
            delta = candidate_score - current_score

            ### Apply Metropolis-Hastings acceptance conditions
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_seq = candidate_seq
                current_score = candidate_score

                #### Keep track of global historical peak
                if current_score > best_score:
                    best_score = current_score
                    best_seq = candidate_seq

            ## Log execution state metrics
            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq,
                "interpreter_weights": importance,
                "temperature": temperature
            })

            ## Cool down the system state
            temperature *= self.cooling_rate

        # Assemble result output map
        result_map = {
            "best_sequence": best_seq,
            "trajectory": trajectory
        }

        if method == "reconstruction":
            predicted = score(best_seq)
            result_map["predicted_activity"] = predicted
            result_map["reconstruction_error"] = abs(predicted - target_expression)
        else:
            result_map["best_score"] = best_score

        return result_map