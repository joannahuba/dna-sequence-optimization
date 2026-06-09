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
        interpretation,
        config
    ):

        method = config.get("method", "optimization")
        mutation_budget = config.get("mutation_budget", None)
        target_expression = config.get("target_expression", None)

        importance = interpretation.importance_scores

        logger.info(
            f"[BeamSearch-SCAN] start | method={method} | "
            f"beam_width={self.beam_width} | top_k={self.top_k_positions}"
        )

        # -------------------------
        # scoring functions
        # -------------------------
        def score(seq):
            result = model_manager.predict_sequences([seq])[seq]
            return sum(result.values()) / len(result)

        def reconstruction_score(seq):
            # return -abs(score(seq) - target_expression)
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

        beam = [sequence]
        best_seq = sequence
        trajectory = []

        # -------------------------
        # adapter detection
        # -------------------------
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

        # -------------------------
        # MAIN LOOP
        # -------------------------
        for it in range(max_iterations):

            candidates = []

            important_positions = MutationGenerator.top_k_positions(
                importance,
                k=self.top_k_positions
            )

            for parent in beam:

                if not self.validator.is_valid(parent):
                    continue

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

            if not candidates:
                logger.warning(
                    f"[BeamSearch-SCAN] STOP iter={it} | no candidates"
                )
                break

            candidates.sort(reverse=True, key=lambda x: x[0])

            beam = [c[1] for c in candidates[:self.beam_width]]

            current_best_score, current_best_seq = candidates[0]

            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq

                logger.info(
                    f"[BeamSearch-SCAN] NEW BEST | iter={it} | score={best_score:.5f}"
                )

            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq
            })

            logger.info(
                f"[BeamSearch-SCAN] iter={it} | best_score={best_score:.5f}"
            )

        # -------------------------
        # OUTPUT
        # -------------------------
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