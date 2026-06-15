# Heading 1 (Interpretation Contract Architecture)
## Structural math backends, model evaluation abstractions, and objective tracking
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch

from ...loss_functions.strategies.base_loss_function import BaseLossObjective
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class BaseAttributionStrategy(ABC):
    """
    Abstract base class defining the functional tensor interface for
    molecular attribution and gradient tracking across networks.
    """

    def __init__(self, objective: BaseLossObjective):
        """
        Initializes the attribution strategy with a unified loss objective.

        :param objective: Injected cost function managing tensor loss and numpy fitness.
        :type objective: BaseLossObjective
        """
        self.objective = objective
        logger.debug("BaseAttributionStrategy initialized with objective: %s", objective.__class__.__name__)

    @abstractmethod
    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.

        :param registered_models: List of available model identifiers inside the active manager.
        :type registered_models: List[str]
        :return: Map tracking output layout directives.
        :rtype: Dict[str, Any]
        """
        pass

    @abstractmethod
    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module,
        metadata: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Computes position-specific importance metrics for an already loaded tensor block.

        :param tensor_x: Pre-allocated encoded sequence tensor matrix shape (B, L, 4).
        :type tensor_x: torch.Tensor
        :param model_instance: Initialized and synchronized PyTorch network object.
        :type model_instance: torch.nn.Module
        :param metadata: Contextual tracking metadata mapping parameter restrictions.
        :type metadata: Dict[str, Any]
        :return: Detached plain CPU-mapped list structure tracking gradient indices.
        :rtype: List[List[float]]
        """
        pass