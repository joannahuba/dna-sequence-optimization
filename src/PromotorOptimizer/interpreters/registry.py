# interpretation/registry.py

from .strategies import (
    BaseAttributionStrategy,
    InSilicoMutagenesis, InSilicoMutagenesisAggregated,
    SaliencyInterpreter, SaliencyInterpreterAggregated,
    IntegratedGradientsStrategy, IntegratedGradientsStrategyAggregated
)
from ..utils.logger import get_custom_logger

# Logger initialization
## Initialize logger using module name as required by logging architecture rules
logger = get_custom_logger(__name__)

# Registry mapping
## Map string identifiers to their respective strategy classes
INTERPRETER_MAP = {
    "saliency": SaliencyInterpreter,
    "saliency_aggregated": SaliencyInterpreterAggregated,
    "integrated_gradients": IntegratedGradientsStrategy,
    "integrated_gradients_aggregated": IntegratedGradientsStrategyAggregated,
    "mutagenesis": InSilicoMutagenesis,
    "mutagenesis_aggregated": InSilicoMutagenesisAggregated
}

class InterpreterRegistry:
    """
    Registry handler for instantiating interpretation strategies.
    """

    @staticmethod
    def load(names: list) -> list:
        """
        Instantiate interpretation strategies based on a list of names.

        :param names: List of identifiers for required strategies.
        :return: List of instantiated strategy objects.
        :raises ValueError: If an identifier is not found in the registry.
        """
        # Result container
        registry = []

        # Iterate through requested names
        for name in names:
            ## Resolve strategy class
            strategy_cls = INTERPRETER_MAP.get(name)

            if strategy_cls is None:
                ### Log error when identifier lookup fails using lazy evaluation
                logger.error("Attempted to load unknown interpreter: %s", name)
                raise ValueError(f"Unknown interpreter: {name}")

            ## Instantiate and append
            logger.info("Instantiating interpreter: %s", name)
            registry.append(strategy_cls())

        return registry