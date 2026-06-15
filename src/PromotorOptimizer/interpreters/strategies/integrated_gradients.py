# Heading 1 (Integrated Gradients Strategy Implementation)
## Operational mathematical backends, tensor integration, and path tracking utilities
import torch
import numpy as np
from typing import Dict, List, Any

from .base_attribution_strategy import BaseAttributionStrategy
from ...loss_functions.strategies.base_loss_function import BaseLossObjective
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class IntegratedGradientsStrategy(BaseAttributionStrategy):
    """
    Executes path integral Riemann integration independently per model 
    driven by a task-specific loss objective.
    """

    def __init__(self, objective: BaseLossObjective, steps: int = 30):
        """
        Initializes the non-aggregated path integral strategy.

        :param objective: Injected cost function managing tensor loss calculations.
        :type objective: BaseLossObjective
        :param steps: The number of Riemann integration steps along the path trajectory.
        :type steps: int
        """
        super().__init__(objective)
        self.steps = steps

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.
        """
        return {
            "is_aggregated": False
        }

    def _generate_gc_matched_baseline(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        Generates a uniform reference baseline matching the structural shape dimensions of the input.
        """
        batch_size, seq_len, channels = tensor_x.shape
        baseline_np = np.full((batch_size, seq_len, channels), 0.25, dtype=np.float32)
        return torch.tensor(baseline_np, dtype=torch.float32, device=tensor_x.device)

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module,
        metadata: Dict[str, Any]
    ) -> List[List[float]]:
        """
        Computes position-specific importance metrics via Riemann path integration of objective gradients.
        """
        if tensor_x.shape[0] > 1:
            raise ValueError("IntegratedGradientsStrategy only supports single-sequence operations.")

        model_instance.eval()
        model_instance.zero_grad()
        
        baseline = self._generate_gc_matched_baseline(tensor_x)
        total_gradients = torch.zeros_like(tensor_x)

        # Path integral accumulation loop
        ## Step iteratively along the linear interpolation path between baseline and input
        for step_idx in range(self.steps):
            alpha = float(step_idx) / float(self.steps)
            interpolated_input = (baseline + alpha * (tensor_x - baseline)).detach()
            interpolated_input.requires_grad_(True)

            ### Execute model pass inside clean gradient computation tracking scopes
            model_instance.zero_grad()
            _, ratio = model_instance(interpolated_input)
            
            ### Compute scalar path target using the injected objective function
            loss = self.objective.evaluate_tensor_loss(predictions=ratio, tensor_x=interpolated_input, metadata=metadata)
            loss.backward()

            ### Capture backpropagated graph parameters from the interpolation nodes
            if interpolated_input.grad is not None:
                total_gradients += interpolated_input.grad

        # Final average and detachment profiling
        ## Apply the fundamental theorem of calculus approximation to compute absolute attributions
        avg_gradients = total_gradients / float(self.steps)
        integrated_attributions = ((tensor_x - baseline) * avg_gradients).detach().abs()
        
        ## Squeeze the batch dimension and serialize matrix coordinates to a list layout on CPU memory
        attributions_list = integrated_attributions[0].cpu().numpy().tolist()
        
        model_instance.zero_grad()
        return attributions_list
    
    
class IntegratedGradientsStrategyAggregated(BaseAttributionStrategy):
    """
    Executes path integral Riemann integration on pre-allocated tensor spaces
    to track position-specific gradient attributions across ensemble configurations.
    """

    def __init__(self, objective: BaseLossObjective, steps: int = 30):
        """
        Initializes the path integral approximation parameters.

        :param objective: Injected cost function managing tensor loss calculations.
        :type objective: BaseLossObjective
        :param steps: The number of Riemann integration steps along the path trajectory.
        :type steps: int
        """
        super().__init__(objective)
        self.steps = steps

    def _generate_gc_matched_baseline(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        Generates a uniform reference baseline matching the structural shape dimensions of the input.
        """
        batch_size, seq_len, channels = tensor_x.shape
        baseline_np = np.full((batch_size, seq_len, channels), 0.25, dtype=np.float32)
        return torch.tensor(baseline_np, dtype=torch.float32, device=tensor_x.device)

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.
        """
        logger.debug("Resolving schema layout options for Integrated Gradients Aggregated Strategy.")
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
        Computes position-specific importance metrics for an already loaded ensemble tensor block.
        """
        if tensor_x.shape[0] > 1:
            logger.error("IntegratedGradientsStrategyAggregated received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("IntegratedGradientsStrategyAggregated only supports single-sequence operations.")

        model_instance.eval()
        model_instance.zero_grad()
        
        baseline = self._generate_gc_matched_baseline(tensor_x)
        total_gradients = torch.zeros_like(tensor_x)

        for step_idx in range(self.steps):
            alpha = float(step_idx) / float(self.steps)
            interpolated_input = (baseline + alpha * (tensor_x - baseline)).detach()
            interpolated_input.requires_grad_(True)

            model_instance.zero_grad()
            _, ratio = model_instance(interpolated_input)
            
            ## Dynamically track the composite loss function path for the ensemble track
            loss = self.objective.evaluate_tensor_loss(predictions=ratio, tensor_x=interpolated_input, metadata=metadata)
            loss.backward()

            if interpolated_input.grad is not None:
                total_gradients += interpolated_input.grad

        avg_gradients = total_gradients / float(self.steps)
        integrated_attributions = ((tensor_x - baseline) * avg_gradients).detach().abs()
        attributions_list = integrated_attributions[0].cpu().numpy().tolist()
        
        model_instance.zero_grad()
        logger.debug("Successfully completed Integrated Gradients Aggregated calculation pass.")
        return attributions_list