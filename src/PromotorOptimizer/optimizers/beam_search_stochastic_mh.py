# Stochastic Beam Search with Metropolis-Hastings tracking
## Import required core libraries and numeric frameworks
import math
import random
import numpy as np
import torch
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
        # REMARK: we need high parameters, on below temperatue 8.5 nothing is changes because gradients are 0 every where
        initial_temperature=0.8,
        cooling_rate=0.985
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
        Executes a vectorized Stochastic Beam Search using the Metropolis-Hastings criteria
        combined with lineage-locked Tabu position masking.
        """
        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)
        iterations = config.get("iterations", 50)
        model_type = config.get("model_type", "ensemble")

        def score(seq):
            result = model_manager.predict_sequences([seq])
            return sum(result[seq].values()) / len(result[seq])

        def reconstruction_score(seq):
            return -abs(score(seq) - target_expression)

        initial_score = reconstruction_score(sequence) if method == "reconstruction" else score(sequence)
        
        beam = [(initial_score, sequence, set())]
        best_seq = sequence
        best_score = initial_score
        temperature = self.initial_temperature
        trajectory = []

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
                except Exception:
                    pass

        # Main optimization iteration loop
        for it in range(iterations):
            candidates = []
            active_sequences = [node[1] for node in beam]

            # Vectorized batch attribution calculation pass
            if hasattr(interpreter, "explain_batch"):
                interpretations = interpreter.explain_batch(model_manager, active_sequences, model_type)
                importance_map = {interp.sequence: interp.importance_scores for interp in interpretations}
            else:
                importance_map = {s: interpreter.explain(model_manager, s, model_type).importance_scores for s in active_sequences}

            # Exploit and mutate step loops over lineage tracks
            for parent_score, parent_seq, mutated_positions in beam:
                raw_importance = importance_map[parent_seq]
                importance_tensor = raw_importance.clone() if hasattr(raw_importance, "clone") else torch.tensor(raw_importance)

                # Zero out importance entries at historically modified coordinates
                for blocked_pos in mutated_positions:
                    importance_tensor[blocked_pos, :] = 0.0

                # TODO REMARK - it should be moved to parameters to be sufficient 
                # Determine correct reduction flag based on interpreter instance type
                reduction_mode = "max" if interpreter.__class__.__name__ == "InSilicoMutagenesis" else "sum"

                
                # redundant 
                # # Apply conditional reduction logic branch
                # if interpreter.__class__.__name__ == "InSilicoMutagenesis":
                #     scores = importance_tensor.max(dim=1)[0].detach().cpu().numpy()
                # else:
                #     scores = importance_tensor.abs().sum(dim=1).detach().cpu().numpy()

                for _ in range(self.candidates_per_parent):
                    child = MutationGenerator.hybrid_mutation(
                        parent_seq, importance_scores=importance_tensor, n_mutations=1,
                        prefix_len=prefix_len, suffix_len=suffix_len, reduction_mode=reduction_mode
                    )
                    
                    if not self.validator.is_valid(child):
                        continue
                        
                    child_score = reconstruction_score(child) if method == "reconstruction" else score(child)
                    delta = child_score - parent_score
                    
                    changed_idx = [i for i in range(len(parent_seq)) if parent_seq[i] != child[i]]
                    new_mutations = mutated_positions | set(changed_idx)

                    if delta > 0 or random.random() < math.exp(delta / temperature):
                        candidates.append((child_score, child, new_mutations))
                    else:
                        candidates.append((parent_score, parent_seq, mutated_positions))
            
            if not candidates:
                break
                
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            unique_candidates = []
            seen_seqs = set()
            for s, seq_str, mut_set in candidates:
                if seq_str not in seen_seqs:
                    seen_seqs.add(seq_str)
                    unique_candidates.append((s, seq_str, mut_set))
                    if len(unique_candidates) == self.beam_width:
                        break
                        
            beam = unique_candidates
            current_best_score, current_best_seq, _ = beam[0]
            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq
                
            # Safely transform matrix to list architecture
            best_importance = importance_map.get(best_seq, list(importance_map.values())[0])
            if hasattr(best_importance, "detach"):
                importance_log = best_importance.detach().cpu().numpy().tolist()
            else:
                importance_log = np.array(best_importance).tolist()

            ## Compile data maps for the entire active population inside the beam
            ### Capturing full lineage tracking states to audit ensemble behavior post-execution
            beam_population_log = [
                {
                    "score": float(node_score),
                    "sequence": node_seq,
                    "mutated_positions": list(node_mut_set)
                }
                for node_score, node_seq, node_mut_set in beam
            ]

            # Log execution metrics for trajectory analysis
            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq,
                "beam_population": beam_population_log,
                "interpreter_weights": importance_log,
                "temperature": temperature
            })
            temperature *= self.cooling_rate

        result = {"best_sequence": best_seq, "trajectory": trajectory}
        if method == "reconstruction":
            predicted = score(best_seq)
            result["predicted_activity"] = predicted
            result["reconstruction_error"] = abs(predicted - target_expression)
        else:
            result["best_score"] = best_score

        return result