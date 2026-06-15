# Heading 1 (Saliency Attribution Strategy Implementation)
## Operational mathematical backends, gradient tracking, and tensor decomposition tools
import torch
from typing import Dict, List, Any

from .base_attribution_strategy import BaseAttributionStrategy
from ...loss_functions.strategies.base_loss_function import BaseLossObjective
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class SaliencyInterpreter(BaseAttributionStrategy):
    """
    Computes instantaneous vanilla saliency attribution maps for independent, 
    discrete models driven by a task-specific loss objective[cite: 18].
    """  

    def __init__(self, objective: BaseLossObjective):
        """
        Initializes the discrete saliency interpretation strategy[cite: 18].

        :param objective: Injected cost function managing tensor loss calculations[cite: 18].
        :type objective: BaseLossObjective
        """
        super().__init__(objective)
        logger.info("Discrete SaliencyInterpreter strategy instantiated successfully.")

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles[cite: 18].
        """
        logger.debug("Resolving discrete schema layout options for Saliency Strategy.")
        return {
            "is_aggregated": False
        }

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module,
        metadata: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Computes position-specific saliency importance metrics driven by the objective function[cite: 18].
        """
        if tensor_x.shape[0] > 1:
            logger.error("Discrete SaliencyInterpreter received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("SaliencyInterpreter only supports single-sequence operations.")

        # Infrastructure synchronization
        ## Ensure model tracking parameters are set to evaluation mode[cite: 18]
        model_instance.eval()
        model_instance.zero_grad()

        # Gradient computational graph attachment
        ## Clone the input tensor target and declare gradient collection requirements[cite: 18]
        x_req = tensor_x.clone().detach().requires_grad_(True)

        # Forward execution and objective evaluation
        ## Compute network inference predictions over the active sequence tensor channel[cite: 18]
        _, ratio = model_instance(x_req)
        
        ## Compute differentiable loss target from the injected objective layer[cite: 18]
        loss = self.objective.evaluate_tensor_loss(predictions=ratio, tensor_x=x_req, metadata=metadata)

        ## Execute the backward graph tracking sequence pass on the computed loss function[cite: 18]
        model_instance.zero_grad()
        loss.backward()

        # Attribution matrix extraction
        ## Isolate absolute values of gradients to capture local feature sensitivity profiles[cite: 18]
        grad = x_req.grad
        if grad is None:
            logger.error("Failed to capture backpropagated gradient tensor components from graph execution.")
            raise RuntimeError("Gradient computation failed during Saliency matrix backpropagation.")

        saliency_matrix = grad.abs()
        
        ## Strip batch configurations and map the multi-channel matrix directly to a host list layout[cite: 18]
        attributions_list = saliency_matrix[0].cpu().numpy().tolist()

        model_instance.zero_grad()
        logger.debug("Successfully completed Saliency interpretation matrix extraction pass.")
        return attributions_list


class SaliencyInterpreterAggregated(BaseAttributionStrategy):
    """
    Computes instantaneous vanilla saliency attribution maps for aggregated ensemble execution profiles[cite: 18].
    """

    def __init__(self, objective: BaseLossObjective):
        """
        Initializes the saliency interpretation strategy.

        :param objective: Injected cost function managing tensor loss calculations.
        :type objective: BaseLossObjective
        """
        super().__init__(objective)
        logger.info("SaliencyInterpreterAggregated strategy instantiated successfully.")

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.

        :param registered_models: List of available model identifiers inside the active manager.
        :type registered_models: List[str]
        :return: Map tracking output layout directives (is_aggregated flag status).
        :rtype: Dict[str, Any]
        """
        logger.debug("Resolving schema layout options for Saliency Aggregated Strategy.")
        return {
            "is_aggregated": True
        }

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module,
        metadata: Dict[str, Any]
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
            logger.error("SaliencyInterpreterAggregated received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("SaliencyInterpreterAggregated only supports single-sequence operations.")

        model_instance.eval()
        model_instance.zero_grad()

        x_req = tensor_x.clone().detach().requires_grad_(True)

        _, ratio = model_instance(x_req)
        
        ## Apply unified objective function to evaluate ensemble element loss
        loss = self.objective.evaluate_tensor_loss(predictions=ratio, tensor_x=x_req, metadata=metadata)

        model_instance.zero_grad()
        loss.backward()

        # Attribution matrix extraction
        ## Isolate absolute values of gradients to capture local feature sensitivity profiles
        grad = x_req.grad
        if grad is None:
            logger.error("Failed to capture backpropagated gradient components from aggregated graph execution.")
            raise RuntimeError("Gradient computation failed during Aggregated Saliency backpropagation.")

        saliency_matrix = grad.abs()
        
        ## Strip batch configurations and map the multi-channel matrix directly to a host list layout
        attributions_list = saliency_matrix[0].cpu().numpy().tolist()

        model_instance.zero_grad()
        logger.debug("Successfully completed Saliency interpretation matrix extraction pass.")
        return attributions_list