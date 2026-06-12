# Heading 1 (Saliency Attribution Strategy Implementation)
## Operational mathematical backends, gradient tracking, and tensor decomposition tools
import torch
from typing import Dict, List, Any

from .base_attribution_strategy import BaseAttributionStrategy
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class SaliencyInterpreter(BaseAttributionStrategy):
    """
    Computes instantaneous vanilla saliency attribution maps by calculating 
    the absolute value of the gradients with respect to the input tensor.
    """

    def __init__(self, is_aggregated: bool = False):
        """
        Initializes the saliency interpretation strategy.

        :param is_aggregated: Flag determining if the strategy aggregates multi-model responses.
        :type is_aggregated: bool
        """
        self.is_aggregated = is_aggregated
        logger.info("SaliencyInterpreter strategy instantiated successfully.")

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.

        :param registered_models: List of available model identifiers inside the active manager[cite: 9].
        :type registered_models: List[str]
        :return: Map tracking output layout directives (is_aggregated flag status).
        :rtype: Dict[str, Any]
        """
        logger.debug("Resolving schema layout options for Saliency Strategy.")
        return {
            "is_aggregated": self.is_aggregated
        }

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module
    ) -> List[List[float]]:
        """
        Computes position-specific saliency importance metrics for an already loaded tensor block.

        :param tensor_x: Pre-allocated encoded sequence tensor matrix shape (1, L, 4).
        :type tensor_x: torch.Tensor
        :param model_instance: Initialized and synchronized PyTorch network object.
        :type model_instance: torch.nn.Module
        :return: Detached plain CPU-mapped list structure tracking gradient indices.
        :rtype: List[List[float]]
        """
        if tensor_x.shape[0] > 1:
            logger.error("SaliencyInterpreter received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("SaliencyInterpreter only supports single-sequence operations.")

        # Infrastructure synchronization
        ## Ensure model tracking parameters are set to evaluation mode
        model_instance.eval()
        model_instance.zero_grad()

        # Gradient computational graph attachment
        ## Clone the input tensor target and declare gradient collection requirements
        x_req = tensor_x.clone().detach().requires_grad_(True)

        # Forward execution and error backpropagation
        ## Compute network inference predictions over the active sequence tensor channel
        _, ratio = model_instance(x_req)
        score = ratio.mean()

        ## Execute the backward graph tracking sequence pass
        model_instance.zero_grad()
        score.backward()

        # Attribution matrix extraction
        ## Isolate absolute values of gradients to capture local feature sensitivity profiles
        grad = x_req.grad
        if grad is None:
            logger.error("Failed to capture backpropagated gradient tensor components from graph execution.")
            raise RuntimeError("Gradient computation failed during Saliency matrix backpropagation.")

        saliency_matrix = grad.abs()
        
        ## Strip batch configurations and map the multi-channel matrix directly to a host list layout
        attributions_list = saliency_matrix[0].cpu().numpy().tolist()

        model_instance.zero_grad()
        logger.debug("Successfully completed Saliency interpretation matrix extraction pass.")
        return attributions_list