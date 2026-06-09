# Stochastic Beam Search with Metropolis-Hastings tracking
## Import required core libraries and numeric frameworks
import math
import random
import numpy as np
from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .validator import SequenceValidator


class StochasticBeamSearchMetropolis(BaseOptimizer):
    """
    Stochastic Beam Search optimizer utilizing the Metropolis-Hastings criterion.

    This architecture avoids premature population convergence (beam collapse) 
    by introducing temperature-dependent probabilistic acceptance for candidate 
    sequences that exhibit lower fitness levels compared to their direct progenitors.

    :param validation_config: Dictionary containing baseline regulatory constraint parameters.
    :type validation_config: dict
    :param beam_width: The tracking pool footprint size maintained per iteration step. Default is 30.
    :type beam_width: int
    :param candidates_per_parent: Mutation variants spawned per parent node within a single step. Default is 10.
    :type candidates_per_parent: int
    :param initial_temperature: Starting thermal variable for Metropolis evaluation loops. Default is 5.0.
    :type initial_temperature: float
    :param cooling_rate: Multiplicative geometric decay factor applied to the system temperature. Default is 0.95.
    :type cooling_rate: float
    """

    def __init__(
        self,
        validation_config,
        beam_width=30,
        candidates_per_parent=10,
        initial_temperature=5.0,
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
        Runs the stochastic multi-trajectory search to optimize the biological sequence.

        This method dynamically re-computes position importance scores for each active 
        parent sequence inside the beam during every iteration loop step.

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

        # Sequence adapter boundaries evaluation
        ## Set up prefix and suffix tracking metrics for sequence isolation
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
        
        # Beam pool elements store tuples matching: (fitness_score, sequence_string)
        beam = [(initial_score, sequence)]
        best_seq = sequence
        best_score = initial_score
        
        temperature = self.initial_temperature
        trajectory = []

        # Main optimization iteration loop
        for it in range(iterations):
            ## Initialize empty candidate generation pool for current cycle
            candidates = []
            
            ## Generate and evaluate children for each sequence in the current beam
            for parent_score, parent_seq in beam:
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
                    
                    ### Verify structural and biological criteria
                    if not self.validator.is_valid(child):
                        continue
                        
                    ### Compute the ensemble fitness score for the child
                    child_score = reconstruction_score(child) if method == "reconstruction" else score(child)
                    
                    ### Evaluate transition using the Metropolis-Hastings criterion
                    delta = child_score - parent_score
                    
                    if delta > 0 or random.random() < math.exp(delta / temperature):
                        candidates.append((child_score, child))
                    else:
                        candidates.append((parent_score, parent_seq))
            
            ## Handle edge case where no candidates are present
            if not candidates:
                break
                
            ## Rank and filter the next generation beam pool
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            ## Unique sequence deduplication to prevent beam collapse
            unique_candidates = []
            seen_seqs = set()
            for s, seq_str in candidates:
                ### Filter out redundant sequence variations
                if seq_str not in seen_seqs:
                    seen_seqs.add(seq_str)
                    unique_candidates.append((s, seq_str))
                    if len(unique_candidates) == self.beam_width:
                        break
                        
            beam = unique_candidates
            
            ## Update global historical peak configurations
            current_best_score, current_best_seq = beam[0]
            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq
                
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