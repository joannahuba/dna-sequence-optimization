# Heading 1 (Interpreter Registry Architecture)

## Core strategy mappings, module factory loading protocols, and objective injections

from typing import List, Any

from .strategies import (
    BaseAttributionStrategy,
    InSilicoMutagenesis, InSilicoMutagenesisAggregated,
    SaliencyInterpreter, SaliencyInterpreterAggregated,
    IntegratedGradientsStrategy, IntegratedGradientsStrategyAggregated
)
from ..loss_functions.strategies.base_loss_function import BaseLossObjective
from ..utils.logger import get_custom_logger

# Instantiation Protocol

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
    Registry handler for instantiating interpretation strategies with injected objectives.
    """
    @staticmethod
    def load(names: List[str], objective: BaseLossObjective) -> List[Any]:
        """
        Instantiate interpretation strategies based on a list of names and inject the objective function.

        :param names: List of identifiers for required strategies.
        :type names: List[str]
        :param objective: Centralized loss objective contract to inject into each interpreter instance.
        :type objective: BaseLossObjective
        :return: List of instantiated strategy objects.
        :rtype: List[Any]
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

            ## Instantiate with objective injection and append to tracking array
            logger.info("Instantiating interpreter: %s", name)
            registry.append(strategy_cls(objective=objective))

        return registry
