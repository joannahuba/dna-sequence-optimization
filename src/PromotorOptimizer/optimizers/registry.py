# optimizers/registry.py

from ..utils.logger import get_custom_logger
from .strategies import (
    BeamSearchOptimizer,
    StochasticMetropolisOptimizer,
    StochasticBeamSearchBoltzmann,
)

# Initialize logger
logger = get_custom_logger(__name__)

# Registry map definition
## Centralized mapping of strings to callable strategies
OPTIMIZER_MAP = {
    "beam_search": BeamSearchOptimizer,
    "beam_stochastic_mh": StochasticMetropolisOptimizer,
    "beam_search_stochastic_boltzman": StochasticBeamSearchBoltzmann
}

class OptimizerRegistry:
    """
    Registry class to handle instantiation of optimization strategies.
    """

    @staticmethod
    def load(names: list, validation_config: dict = None) -> list:
        """
        Load and instantiate optimizers based on the provided names.

        :param names: A list of string identifiers for the required optimizers.
        :param validation_config: Configuration dictionary for the optimizers.
        :return: A list of instantiated optimizer objects.
        :raises ValueError: If a requested optimizer name is not defined in OPTIMIZER_MAP.
        """
        # Result container
        registry = []

        # Iterate and instantiate
        for name in names:
            ## Retrieve factory from mapping
            optimizer_factory = OPTIMIZER_MAP.get(name)

            if optimizer_factory is None:
                logger.error("Attempted to load unregistered optimizer: %s", name)
                raise ValueError(f"Unknown optimizer: {name}")

            ## Instantiate with configuration
            logger.info("Initializing optimizer instance: %s", name)
            registry.append(optimizer_factory(validation_config))

        return registry