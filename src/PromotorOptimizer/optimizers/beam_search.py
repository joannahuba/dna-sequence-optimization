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

    def _score(
        self,
        model_manager,
        sequence
    ):

        result = model_manager.predict_sequences(
            [sequence]
        )

        scores = list(
            result[sequence].values()
        )

        return sum(scores) / len(scores)

    def _reconstruction_score(
        self,
        model_manager,
        sequence,
        target_expression
    ):
        """
        Higher is better.
        Best value = 0.
        """

        predicted = self._score(
            model_manager,
            sequence
        )

        return -abs(
            predicted - target_expression
        )

    def optimize(
        self,
        sequence,
        model_manager,
        interpretation,
        config
    ):

        method = config.get(
            "method",
            "optimization"
        )

        mutation_budget = config.get(
            "mutation_n",
            None
        )

        target_expression = config.get(
            "org_expression",
            None
        )

        importance = (
            interpretation.importance_scores
        )

        logger.info(
            "[BeamSearch] Starting optimization "
            f"(mode={method}, sequence_length={len(sequence)})"
        )

        if method == "reconstruction":
            logger.info(
                "[BeamSearch] Reconstruction target="
                f"{target_expression:.4f}, "
                f"mutation_budget={mutation_budget}"
            )

        beam = [sequence]

        trajectory = []

        if method == "reconstruction":

            best_score = self._reconstruction_score(
                model_manager,
                sequence,
                target_expression
            )

            logger.info(
                "[BeamSearch] Initial reconstruction score=%.6f",
                best_score
            )

        else:

            best_score = self._score(
                model_manager,
                sequence
            )

            logger.info(
                "[BeamSearch] Initial activity score=%.6f",
                best_score
            )

        best_seq = sequence

        max_iterations = (
            mutation_budget
            if (
                method == "reconstruction"
                and mutation_budget is not None
            )
            else self.iterations
        )

        logger.info(
            "[BeamSearch] Running for %d iterations",
            max_iterations
        )

        for iteration in range(max_iterations):

            candidates = []

            invalid_count = 0

            for parent in beam:

                if not self.validator.is_valid(
                        parent
                    ):
                    print("parent is not valid")
                for _ in range(
                    self.candidates_per_parent
                ):

                    child = (
                        MutationGenerator.hybrid_mutation(
                            parent,
                            importance,
                            n_mutations=1,
                            lambda_weight=0.8
                        )
                    )

                    if not self.validator.is_valid(
                        child
                    ):
                        invalid_count += 1
                        continue

                    if method == "reconstruction":

                        score = (
                            self._reconstruction_score(
                                model_manager,
                                child,
                                target_expression
                            )
                        )

                    else:

                        score = self._score(
                            model_manager,
                            child
                        )

                    candidates.append(
                        (score, child)
                    )

            if not candidates:

                logger.warning(
                    "[BeamSearch] Iteration %d produced "
                    "no valid candidates. Stopping.",
                    iteration
                )

                break

            beam = [
                seq
                for _, seq in heapq.nlargest(
                    self.beam_width,
                    candidates
                )
            ]

            current_best_score, current_best_seq = max(
                candidates,
                key=lambda x: x[0]
            )

            if current_best_score > best_score:

                improvement = (
                    current_best_score
                    - best_score
                )

                best_score = current_best_score
                best_seq = current_best_seq

                logger.info(
                    "[BeamSearch] Iteration %d: "
                    "new best score %.6f "
                    "(improvement %.6f)",
                    iteration,
                    best_score,
                    improvement
                )

            logger.info(
                "[BeamSearch] Iteration %d | "
                "valid=%d | invalid=%d | "
                "best=%.6f",
                iteration,
                len(candidates),
                invalid_count,
                best_score
            )

            trajectory.append(
                {
                    "iteration": iteration,
                    "sequence": best_seq,
                    "score": float(best_score),
                    "valid": True
                }
            )

        logger.info(
            "[BeamSearch] Search finished. "
            "Best score=%.6f",
            best_score
        )

        result = {
            "best_sequence": best_seq,
            "trajectory": trajectory
        }

        if method == "reconstruction":

            predicted_activity = self._score(
                model_manager,
                best_seq
            )

            reconstruction_error = abs(
                predicted_activity
                - target_expression
            )

            logger.info(
                "[BeamSearch] Reconstruction complete | "
                "target=%.6f | predicted=%.6f | "
                "error=%.6f",
                target_expression,
                predicted_activity,
                reconstruction_error
            )

            result["reconstruction_error"] = (
                reconstruction_error
            )

            result["predicted_activity"] = (
                predicted_activity
            )

        else:

            logger.info(
                "[BeamSearch] Optimization complete | "
                "best_activity=%.6f",
                best_score
            )

            result["best_score"] = best_score

        return result