import heapq
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
        # scoring
        # -------------------------
        def score(seq):
            result = model_manager.predict_sequences([seq])
            return sum(result[seq].values()) / len(result[seq])

        def reconstruction_score(seq):
            return -abs(score(seq) - target_expression)

        # -------------------------
        # init
        # -------------------------
        beam = [sequence]
        best_seq = sequence

        if method == "reconstruction":
            best_score = reconstruction_score(sequence)
            max_iterations = mutation_budget  # ✅ FIX
        else:
            best_score = score(sequence)
            max_iterations = self.iterations

        trajectory = []

        print(f"[BeamSearch] mode={method} iterations={max_iterations}")

        # -------------------------
        # main loop
        # -------------------------
        for it in range(max_iterations):

            candidates = []

            for parent in beam:

                if not self.validator.is_valid(parent):
                    continue

                for _ in range(self.candidates_per_parent):

                    # IMPORTANT: keep small mutation step
                    child = MutationGenerator.hybrid_mutation(
                        parent,
                        importance,
                        n_mutations=1,
                        lambda_weight=0.8
                    )

                    if not self.validator.is_valid(child):
                        continue

                    s = reconstruction_score(child) if method == "reconstruction" else score(child)
                    candidates.append((s, child))

            if not candidates:
                print(f"[BeamSearch] STOP iter={it} (no valid candidates)")
                break

            candidates.sort(reverse=True, key=lambda x: x[0])

            beam = [c[1] for c in candidates[:self.beam_width]]

            current_best_score, current_best_seq = candidates[0]

            if current_best_score > best_score:
                best_score = current_best_score
                best_seq = current_best_seq

            trajectory.append({
                "iteration": it,
                "score": float(best_score),
                "sequence": best_seq
            })

            print(f"[BeamSearch] iter={it} best={best_score:.5f}")

        # -------------------------
        # output
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