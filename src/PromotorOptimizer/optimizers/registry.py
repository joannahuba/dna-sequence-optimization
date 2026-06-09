# optimizers/registry.py

from .beam_search import BeamSearchOptimizer
from .simulated_annealing import SimulatedAnnealingOptimizer # Add import line


class OptimizerRegistry:

    @staticmethod
    def load(names, validation_config=None):
        registry = []

        for name in names:
            if name == "beam_search":
                print(validation_config)
                registry.append(
                    BeamSearchOptimizer(validation_config)
                )
            elif name == "simulated_annealing": # Add mapping branch
                registry.append(SimulatedAnnealingOptimizer(validation_config))
            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry