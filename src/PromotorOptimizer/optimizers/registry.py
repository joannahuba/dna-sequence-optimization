# optimizers/registry.py

from .random_mutation_optimizer import RandomMutationOptimizer


class OptimizerRegistry:

    @staticmethod
    def load(names: list):

        registry = []

        for name in names:

            if name == "mutation":
                registry.append(RandomMutationOptimizer())

            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry