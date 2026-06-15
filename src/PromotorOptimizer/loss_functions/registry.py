# Heading 1 (Loss Functions Registry Architecture)
## Centralized objective function registration and instantiation handlers
from typing import List, Dict, Any, Optional
from ..utils.logger import get_custom_logger

# Explicit imports of standalone functional objectives
from .strategies.base_loss_function import BaseLossObjective
from .strategies.reconstruction_loss_basic import ReconstructionLossObjective 
from .strategies.optimization_loss_basic import OptimizationLossObjective

# Instantiation Protocol
logger = get_custom_logger(__name__)

# Registry map definition
## Centralized mapping of strings to callable objective class constructors
OBJECTIVE_MAP = {
    "reconstruction_base": ReconstructionLossObjective,
    "maximization_base": OptimizationLossObjective
}


class ObjectiveRegistry:
    """
    Registry class to handle initialization and injection of pipeline cost profiles.
    """

    @staticmethod
    def load(name: str, objective_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Loads and instantiates a specific loss objective strategy based on its unique string name.

        :param name: String identifier for the required cost function target.
        :type name: str
        :param objective_config: Configuration dictionary passed directly to the constructor.
        :type objective_config: Optional[Dict[str, Any]]
        :return: An instantiated standalone loss objective object.
        :rtype: Any
        :raises ValueError: If the requested objective name is not defined in OBJECTIVE_MAP.
        """
        objective_config = objective_config or {}
        
        ## Retrieve factory constructor from mapping
        objective_class = OBJECTIVE_MAP.get(name)

        if objective_class is None:
            logger.error("Attempted to load unregistered loss objective: %s", name)
            raise ValueError(f"Unknown loss objective configuration target: {name}")

        logger.info("Initializing loss objective interface instance: %s", name)
        return objective_class(objective_config)