import logging

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
        Executes the position-scanning Beam Search optimization tracking loop.

        This method dynamically re-computes importance attribution scores per sequence 
        in the beam during each iteration to handle epistatic regulatory changes.

        :param sequence: Seed wild-type or disrupted DNA sequence string.
        :type sequence: str
        :param model_manager: Unified evaluation suite manager coordination stack.
        :type model_manager: ModelManager
        :param interpreter: Live implementation instance of BaseInterpreter to re-evaluate gradients.
        :type interpreter: BaseInterpreter
        :param config: Execution properties map containing runtime limits and objectives.
        :type config: dict
        :return: Results map detailing the optimal sequence and iteration logs.
        :rtype: dict
        """
        # Configuration parsing and parameter setup
        ## Extract operational modalities from the configuration map
        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)
        model_type = config.get("model_type", "ensemble")

        logger.info(
            f"[BeamSearch-SCAN] start | method={method} | "
            f"beam_width={self.beam_width} | top_k={self.top_k_positions}"
        )

        # Performance evaluation functions
        ## Define raw ensemble prediction scoring mechanics
        def score(seq):
            result = model_manager.predict_sequences([seq])[seq]
            return sum(result.values()) / len(result)

        ## Define alternative tracking for target reconstruction tasks
        def reconstruction_score(seq):
            # return -abs(score(seq) - target_expression)
            result = model_manager.predict_sequences([seq])[seq]
            return sum(result.values()) / len(result)

        ## Assign target evaluation functions and boundary maximum iteration bounds
        if method == "reconstruction":
            candidate_score_fn = reconstruction_score
            best_score = reconstruction_score(sequence)
            max_iterations = mutation_budget
        else:
            candidate_score_fn = score
            best_score = score(sequence)
            max_iterations = self.iterations

        beam = [sequence]
        best_seq = sequence
        trajectory = []

        # Adapter domain detection boundaries
        ## Analyze integrated tracking fields to isolate constant sub-sequences
        prefix_len = 0
        suffix_len = 0

        for model_meta in model_manager.get_models().values():
            dataset_class = model_meta.get("dataset_class")

            if dataset_class and dataset_class.__name__ == "DNADatasetNoAdapters":
                try:
                    input_file = config.get(
                        "input_path",
                        "data/reconstruction_input.tsv"
                    )

                    temp_dataset = dataset_class(input_file)

                    prefix_len = getattr(temp_dataset, "prefix_len", 0)
                    suffix_len = getattr(temp_dataset, "suffix_len", 0)

                    logger.info(
                        f"[BeamSearch-SCAN] adapter detected | "
                        f"prefix_len={prefix_len}, suffix_len={suffix_len}"
                    )
                    break

                except Exception as e:
                    logger.warning(
                        f"[BeamSearch-SCAN] adapter detection failed: {e}"
                    )

        # Evolutionary beam search loop execution
        ## Iterate over calculated sequence modification tracks
        for it in range(max_iterations):

            candidates = []

            ## Scan over elements currently retained within the beam population
            for parent in beam:

                ### Short-circuit check against structural constraints
                if not self.validator.is_valid(parent):
                    continue

                ### Compute live position sensitivity profiles dynamically for the current parent state
                interpretation = interpreter.explain(
                    model_manager=model_manager,
                    sequence=parent,
                    model_type=model_type
                )
                importance = interpretation.importance_scores

                ### Extract highly reactive sequence coordinates based on updated importance maps
                important_positions = MutationGenerator.top_k_positions(
                    importance,
                    k=self.top_k_positions
                )

                ### Sweep point mutations across optimal coordinate targets
                for pos in important_positions:

                    candidates.extend(
                        self._scan_position(
                            parent,
                            int(pos),
                            model_manager,
                            candidate_score_fn,
                            prefix_len=prefix_len,
                            suffix_len=suffix_len
                        )
                    )

            ## Pool verification and empty set monitoring
            if not candidates:
                logger.warning(
                    f"[BeamSearch-SCAN] STOP iter={it} | no candidates"
                )
                break

            ## Sort candidates population to partition the next tracking iteration space
            candidates.sort(reverse=True, key=lambda x: x[0])

            beam = [c[1] for c in candidates[:self.beam_width]]

            current_best_score, current_best_seq = candidates[0]

            ## Update global historical peak configurations
            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq

                logger.info(
                    f"[BeamSearch-SCAN] NEW BEST | iter={it} | score={best_score:.5f}"
                )

            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq,
                "interpreter_weights": importance
            })

            logger.info(
                f"[BeamSearch-SCAN] iter={it} | best_score={best_score:.5f}"
            )

        # Output payload compilation
        ## Package optimal metrics and tracking history maps
        result = {
            "best_sequence": best_seq,
            "trajectory": trajectory
        }

        if method == "reconstruction":
            predicted = score(best_seq)
            result["predicted_activity"] = predicted
            result["reconstruction_error"] = abs(predicted - target_expression)
            

            logger.info(
                f"[BeamSearch-SCAN] finished reconstruction | "
                f"pred={predicted:.5f} | error={result['reconstruction_error']:.5f}"
            )
        else:
            result["best_score"] = best_score

            logger.info(
                f"[BeamSearch-SCAN] finished optimization | "
                f"best_score={best_score:.5f}"
            )

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