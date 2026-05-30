# pipeline/experiment_tracker.py

from typing import List, Dict
from ..core.types import OptimizationStep


class ExperimentTracker:

    def __init__(self):
        self.history: List[OptimizationStep] = []
        self.best_step_cache = None

        print("[Tracker] Initialized")

    def log_step(self, step: OptimizationStep):
        self.history.append(step)

        print(
            f"[Tracker] step={step.iteration} "
            f"score={step.ensemble_score:.4f} "
            f"seq_len={len(step.sequence)}"
        )

        if (
            self.best_step_cache is None
            or step.ensemble_score > self.best_step_cache.ensemble_score
        ):
            self.best_step_cache = step

            print(
                f"[Tracker] NEW BEST → iter={step.iteration} "
                f"score={step.ensemble_score:.4f}"
            )

    def best_step(self):
        return self.best_step_cache

    def export_curve(self):

        print(f"[Tracker] exporting curve: {len(self.history)} steps")

        return [
            {
                "iteration": s.iteration,
                "ensemble_score": s.ensemble_score
            }
            for s in self.history
        ]

    def detect_model_conflicts(self, threshold: float = 0.5):

        print(f"[Tracker] checking conflicts (threshold={threshold})")

        conflicts = []

        for step in self.history:

            scores = list(step.model_scores.values())
            if len(scores) < 2:
                continue

            spread = max(scores) - min(scores)

            if spread >= threshold:
                print(
                    f"[Tracker] conflict iter={step.iteration} "
                    f"spread={spread:.4f}"
                )

                conflicts.append({
                    "iteration": step.iteration,
                    "sequence": step.sequence,
                    "spread": spread,
                    "scores": step.model_scores
                })

        print(f"[Tracker] conflicts found: {len(conflicts)}")

        return conflicts