# optimizers/registry.py

from .beam_search import BeamSearchOptimizer
from .stochastic_mh import SimulatedAnnealingOptimizer # Add import line
from .beam_search_stochastic_mh import StochasticBeamSearchMetropolis
from .beam_search_stochastic_boltzman import StochasticBeamSearchBoltzmann

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
            elif name == "search_stochastic_mh": # Add mapping branch
                registry.append(SimulatedAnnealingOptimizer(validation_config))
            elif name == "beam_search_stochastic_mh": # Add mapping branch
                registry.append(StochasticBeamSearchMetropolis(validation_config))
            elif name == "beam_search_stochastic_boltzman": # Add mapping branch
                registry.append(StochasticBeamSearchBoltzmann(validation_config))
            else:
                raise ValueError(f"Unknown optimizer {name}")

        return registry