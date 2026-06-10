import logging
import numpy as np

from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .validator import SequenceValidator

logger = logging.getLogger(__name__)

class BeamSearchOptimizer(BaseOptimizer):

    def __init__(
        self,
        validation_config,
        beam_width=40,
        top_k_positions=40,
        iterations=50,
    ):
        self.validator = SequenceValidator(validation_config)
        self.beam_width = beam_width
        self.top_k_positions = top_k_positions
        self.iterations = iterations

    def optimize(
        self,
        sequence,
        model_manager,
        interpreter,
        config
    ):
        """
        Executes a vectorized, lineage-preserved position-scanning Beam Search.

        This architecture forces dynamic batch-wise importance re-computation while
        tracking historical mutation coordinates to eliminate cyclical back-tracking.

        :param sequence: Seed wild-type or disrupted DNA sequence string.
        :type sequence: str
        :param model_manager: Unified evaluation suite manager coordination stack.
        :type model_manager: ModelManager
        :param interpreter: Live implementation instance of BaseInterpreter.
        :type interpreter: BaseInterpreter
        :param config: Execution properties map containing runtime variables.
        :type config: dict
        :return: Results map detailing the optimal sequence and iteration logs.
        :rtype: dict
        """
        # Configuration parsing and context initialization
        ## Extract processing targets and iteration constraints
        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)
        model_type = config.get("model_type", "ensemble")

        logger.info(f"[BeamSearch-SCAN] Vectorized execution started | method={method}")

        # Performance evaluation functions
        def score(seq):
            result = model_manager.predict_sequences([seq])[seq]
            return sum(result.values()) / len(result)

        def reconstruction_score(seq):
            result = model_manager.predict_sequences([seq])[seq]
            return sum(result.values()) / len(result)

        if method == "reconstruction":
            candidate_score_fn = reconstruction_score
            best_score = reconstruction_score(sequence)
            max_iterations = mutation_budget
        else:
            candidate_score_fn = score
            best_score = score(sequence)
            max_iterations = self.iterations

        # State tracking initialization
        ## Beam entries store triples matching: (fitness_score, sequence_string, mutated_positions_set)
        beam = [(best_score, sequence, set())]
        best_seq = sequence
        trajectory = []

        # Adapter domain boundaries extraction
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

        # Master evolutionary search loop
        for it in range(max_iterations):
            logger.info(f"[BeamSearch-SCAN] Iteration {it}/{max_iterations} started | Active Beam Size = {len(beam)}")
            candidates = []
            active_sequences = [node[1] for node in beam]

            # Vectorized importance evaluation pass
            ## Compute sensitivity matrices for the entire beam in one parallel GPU execution
            if hasattr(interpreter, "explain_batch"):
                interpretations = interpreter.explain_batch(model_manager, active_sequences, model_type)
                importance_map = {interp.sequence: interp.importance_scores for interp in interpretations}
            else:
                importance_map = {}
                for parent_seq in active_sequences:
                    interp = interpreter.explain(model_manager, parent_seq, model_type)
                    importance_map[parent_seq] = interp.importance_scores

            # Candidate expansion phase over active lineages
            for parent_score, parent_seq, mutated_positions in beam:
                if not self.validator.is_valid(parent_seq):
                    continue

                raw_importance = importance_map[parent_seq]
                importance_tensor = raw_importance.clone() if hasattr(raw_importance, "clone") else torch.tensor(raw_importance)

                # Enforce Tabu tracking by zeroing weights at historically modified coordinates
                for blocked_pos in mutated_positions:
                    importance_tensor[blocked_pos, :] = 0.0

                # TODO REMARK - it should be moved to parameters to be sufficient 
                # Determine correct reduction flag based on interpreter instance type
                reduction_mode = "max" if interpreter.__class__.__name__ == "InSilicoMutagenesis" else "sum"

                # Delegate position extraction to the updated MutationGenerator interface
                important_positions = MutationGenerator.top_k_positions(
                    importance_tensor,
                    k=self.top_k_positions,
                    reduction_mode=reduction_mode
                )

                # Apply conditional matrix reduction depending on interpreter type
                if interpreter.__class__.__name__ == "InSilicoMutagenesis":
                    scores = importance_tensor.max(dim=1)[0].detach().cpu().numpy()
                else:
                    scores = importance_tensor.abs().sum(dim=1).detach().cpu().numpy()

                important_positions = scores.argsort()[::-1][:self.top_k_positions]
                parent_lineage_count = 0

                # Sweep substitutions across selected unmasked coordinates
                for pos in important_positions:
                    if parent_lineage_count >= (self.beam_width // 2):
                        break

                    variants = self._scan_position(
                        parent_seq, int(pos), model_manager, candidate_score_fn, prefix_len, suffix_len
                    )

                    for child_score, child_seq in variants:
                        new_mutations = mutated_positions | {int(pos)}
                        candidates.append((child_score, child_seq, new_mutations))
                        parent_lineage_count += 1

            if not candidates:
                logger.warning(f"[BeamSearch-SCAN] Loop broken at iter {it} | Reason: Candidate pool is empty (all variants rejected by validator constraints)")
                break

            # Selection and deduplication phase
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
            current_best_score, current_best_seq, current_mut_set = beam[0]

            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq

            

            # VRAM & JSON protection extraction
            ## Isolate importance matrix mapped to the step champion sequence context
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
                "temperature": 0.0
            })

            logger.info(f"[BeamSearch-SCAN] iter={it} | best_score={best_score:.5f}")

        # Output dictionary construction
        result = {"best_sequence": best_seq, "trajectory": trajectory}
        if method == "reconstruction":
            predicted = score(best_seq)
            result["predicted_activity"] = predicted
            result["reconstruction_error"] = abs(predicted - target_expression)
        else:
            result["best_score"] = best_score

        return result

    # -------------------------
    # SCAN OPERATOR
    # -------------------------
    def _scan_position(
        self,
        sequence,
        position,
        model_manager,
        score_fn,
        prefix_len: int = 0,
        suffix_len: int = 0
    ):

        BASES = ["A", "C", "G", "T"]

        # adapter protection
        if position < prefix_len:
            return []

        if suffix_len > 0 and position >= len(sequence) - suffix_len:
            return []

        current = sequence[position]
        candidates = []

        for base in BASES:

            if base == current:
                continue

            mutated = list(sequence)
            mutated[position] = base
            mutated = "".join(mutated)

            if not self.validator.is_valid(mutated):
                continue

            fitness = score_fn(mutated)

            candidates.append((fitness, mutated))

        return candidates