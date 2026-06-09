# Stochastic Beam Search with Boltzmann distribution sampling
## Import operational libraries and statistical modules
import math
import random
import numpy as np
import torch
from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .validator import SequenceValidator


class StochasticBeamSearchBoltzmann(BaseOptimizer):
    """
    Stochastic Beam Search optimizer utilizing Boltzmann probability distribution sampling.

    This architecture avoids beam collapse by sampling the next generation pool
    stochastically based on Boltzmann weights computed from ensemble fitness scores.

    :param validation_config: Dictionary containing baseline regulatory constraint parameters.
    :type validation_config: dict
    :param beam_width: The tracking pool footprint size maintained per iteration step. Default is 30.
    :type beam_width: int
    :param candidates_per_parent: Mutation variants spawned per parent node within a single step. Default is 10.
    :type candidates_per_parent: int
    :param initial_temperature: Starting thermal variable scaling the selection distribution. Default is 2.0.
    :type initial_temperature: float
    :param cooling_rate: Multiplicative geometric decay factor applied to the system temperature. Default is 0.95.
    :type cooling_rate: float
    """

    def __init__(
        self,
        validation_config,
        beam_width=30,
        candidates_per_parent=10,
        initial_temperature=2.0,
        cooling_rate=0.95
    ):
        # Operational components setup
        ## Instantiate the structural biological validator interface
        self.validator = SequenceValidator(validation_config)
        self.beam_width = beam_width
        self.candidates_per_parent = candidates_per_parent
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
        Runs the Boltzmann-sampled multi-trajectory search to optimize the biological sequence.

        This method dynamically re-computes position importance scores for each active 
        parent sequence inside the beam during every iteration loop step to adapt to epistatic variations.

        :param sequence: Seed wild-type or disrupted nucleotide string.
        :type sequence: str
        :param model_manager: Unified evaluation engine wrapper orchestration stack.
        :type model_manager: ModelManager
        :param interpreter: Live implementation instance of BaseInterpreter to re-evaluate gradients.
        :type interpreter: BaseInterpreter
        :param config: Runtime configuration block mapping execution goals and mutation limits.
        :type config: dict
        :return: Execution map containing the best candidate sequence and tracking parameters.
        :rtype: dict
        """
        # Configuration parsing and context initialization
        ## Extract processing directives and mutation properties
        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)
        iterations = config.get("iterations", 50)
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

        # Define internal evaluation mechanics
        ## Establish ensemble fitness function calculations
        def score(seq):
            result = model_manager.predict_sequences([seq])
            return sum(result[seq].values()) / len(result[seq])

        ## Establish error tracking for target expression matching tasks
        def reconstruction_score(seq):
            return -abs(score(seq) - target_expression)

        # Optimization variables initialization
        ## Seed initial parameters and active beam pools
        initial_score = reconstruction_score(sequence) if method == "reconstruction" else score(sequence)
        
        beam = [(initial_score, sequence)]
        best_seq = sequence
        best_score = initial_score
        
        temperature = self.initial_temperature
        trajectory = []

        # Main optimization iteration loop
        for it in range(iterations):
            ## Initialize evaluation round variables
            candidates = []
            seen_seqs = set()
            
            ## Collect and score all children across current beam nodes
            for _, parent_seq in beam:
                ### Compute live position sensitivity profiles dynamically for the current parent state
                interpretation = interpreter.explain(
                    model_manager=model_manager,
                    sequence=parent_seq,
                    model_type=model_type
                )
                importance = interpretation.importance_scores
                
                for _ in range(self.candidates_per_parent):
                    ### Generate a single-nucleotide mutation
                    child = MutationGenerator.hybrid_mutation(
                        parent_seq,
                        importance_scores=importance,
                        n_mutations=1,
                        prefix_len=prefix_len,
                        suffix_len=suffix_len
                    )
                    
                    ### Skip duplication to conserve evaluation overhead
                    if child in seen_seqs:
                        continue
                    seen_seqs.add(child)
                    
                    ### Verify structural and biological criteria
                    if not self.validator.is_valid(child):
                        continue
                        
                    ### Compute the ensemble fitness score for the child
                    child_score = reconstruction_score(child) if method == "reconstruction" else score(child)
                    candidates.append((child_score, child))
            
            ## Handle edge case where no candidates survive validation
            if not candidates:
                break
                
            ## Update global historical peak configurations before sampling filter
            candidates.sort(reverse=True, key=lambda x: x[0])
            current_best_score, current_best_seq = candidates[0]
            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq

            ## Boltzmann selection phase
            ### Extract raw scores and normalize to prevent numerical overflow during exponentiation
            raw_scores = np.array([c[0] for c in candidates])
            max_score = np.max(raw_scores)
            
            ### Compute Boltzmann weights adjusted by current system temperature
            with np.errstate(divide='ignore', overflow='ignore'):
                exp_weights = np.exp((raw_scores - max_score) / temperature)
                
            sum_exp = np.sum(exp_weights)
            
            ### Define fallback distribution if division collapses to zero
            if sum_exp == 0 or np.isnan(sum_exp):
                probabilities = np.ones(len(candidates)) / len(candidates)
            else:
                probabilities = exp_weights / sum_exp

            ### Stochastic choice implementation representing selection without replacement
            chosen_indices = np.random.choice(
                len(candidates),
                size=min(self.beam_width, len(candidates)),
                replace=False,
                p=probabilities
            )
            
            ### Build the next iteration beam space
            beam = [candidates[idx] for idx in chosen_indices]
            
            ## Log execution metrics for trajectory analysis
            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq,
                "interpreter_weights": importance,
                "temperature": temperature
            })
            
            ## Apply geometric cooling decay to system temperature
            temperature *= self.cooling_rate

        # Assemble execution tracking results
        ## Compile output dictionary coordinates
        result = {
            "best_sequence": best_seq,
            "trajectory": trajectory
        }

        if method == "reconstruction":
            predicted = score(best_seq)
            result["predicted_activity"] = predicted
            result["reconstruction_error"] = abs(predicted - target_expression)
        else:
            result["best_score"] = best_score

        return result