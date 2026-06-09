import logging

from .base_optimizer import BaseOptimizer
from .mutation_generator import MutationGenerator
from .validator import SequenceValidator

logger = logging.getLogger(__name__)


class BeamSearchOptimizer(BaseOptimizer):

    def __init__(
        self,
        validation_config,
        beam_width=30,
        candidates_per_parent=10,
        iterations=50,
    ):
        self.validator = SequenceValidator(validation_config)
        self.beam_width = beam_width
        self.candidates_per_parent = candidates_per_parent
        self.iterations = iterations

    # -------------------------
    # MAIN OPTIMIZATION
    # -------------------------
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

        # -------------------------
        # unified scoring (IMPORTANT FIX)
        # -------------------------
        def score(seq):
            result = model_manager.predict_sequences([seq])[seq]
            return sum(result.values()) / len(result)

        def reconstruction_score(seq):
            return -abs(score(seq) - target_expression)

        # -------------------------
        # init
        # -------------------------
        beam = [sequence]
        best_seq = sequence

        if method == "reconstruction":
            best_score = reconstruction_score(sequence)
            max_iterations = mutation_budget
        else:
            best_score = score(sequence)
            max_iterations = self.iterations

        trajectory = []

        print(f"[BeamSearch] mode={method} iterations={max_iterations}")

        # -------------------------
        # scoring mode
        # -------------------------
        if method == "reconstruction":
            candidate_score_fn = reconstruction_score
        else:
            candidate_score_fn = score

        # -------------------------
        # MAIN LOOP
        # -------------------------
        for it in range(max_iterations):

            candidates = []

            important_positions = MutationGenerator.top_k_positions(
                importance,
                k=15
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
                            candidate_score_fn
                        )
                    )

            if not candidates:
                print(
                    f"[BeamSearch] STOP iter={it} "
                    "(no valid candidates)"
                )
                break

            candidates.sort(
                reverse=True,
                key=lambda x: x[0]
            )

            beam = [
                c[1]
                for c in candidates[:self.beam_width]
            ]

            current_best_score, current_best_seq = candidates[0]

            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq

            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq
            })

            if method == "reconstruction":

                predicted = score(best_seq)
                error = abs(
                    predicted - target_expression
                )

                print(
                    f"[BeamSearch] iter={it} "
                    f"pred={predicted:.4f} "
                    f"target={target_expression:.4f} "
                    f"error={error:.4f}"
                )

            else:

                print(
                    f"[BeamSearch] iter={it} "
                    f"best={best_score:.5f}"
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
        else:
            result["best_score"] = best_score

        return result

    # -------------------------
    # POSITION SCAN
    # -------------------------
    def _scan_position(
        self,
        sequence,
        position,
        model_manager,
        score_fn
    ):

        BASES = ["A", "C", "G", "T"]
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

            candidates.append(
                (
                    fitness,
                    mutated
                )
            )

        return candidates
    
    # def _scan_position(
    #     self,
    #     sequence,
    #     position,
    #     model_manager,
    #     score_fn
    # ):

    #     BASES = ["A", "C", "G", "T"]
    #     current = sequence[position]

    #     candidates = []

    #     for base in BASES:

    #         if base == current:
    #             continue

    #         mutated = list(sequence)
    #         mutated[position] = base
    #         mutated = "".join(mutated)

    #         if not self.validator.is_valid(mutated):
    #             continue

    #         fitness = score_fn(mutated)

    #         candidates.append(
    #             (
    #                 fitness,
    #                 mutated,
    #                 position,
    #                 current,
    #                 base
    #             )
    #         )

    #     return candidates