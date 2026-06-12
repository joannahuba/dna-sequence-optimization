# Heading 1 (Integrated Gradients Strategy Implementation)
## Operational mathematical backends, tensor integration, and path tracking utilities
import torch
import numpy as np
from typing import Dict, List, Any

from .base import BaseAttributionStrategy
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)

# Heading 1 (Integrated Gradients Discrete Strategy)
## Implement discrete single-model path tracking parameters
class IntegratedGradientsStrategy(BaseAttributionStrategy):
    """
    Executes path integral Riemann integration independently per model 
    without aggregate ensemble projection schemas[cite: 1, 10].
    """

    def __init__(self, steps: int = 30):
        """
        Initializes the non-aggregated path integral strategy[cite: 10].

        :param steps: The number of Riemann integration steps along the path trajectory[cite: 10].
        :type steps: int
        """
        self.steps = steps
        self.is_aggregated = False

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles[cite: 1].

        :param registered_models: List of available model identifiers inside the active manager[cite: 9].
        :type registered_models: List[str]
        :return: Map tracking output layout directives forcing individual model keys[cite: 1].
        :rtype: Dict[str, Any]
        """
        return {
            "is_aggregated": self.is_aggregated
        }

    def _generate_gc_matched_baseline(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        Generates a uniform reference baseline matching the structural shape dimensions of the input[cite: 10].
        """
        batch_size, seq_len, channels = tensor_x.shape[cite: 10]
        baseline_np = np.full((batch_size, seq_len, channels), 0.25, dtype=np.float32)[cite: 10]
        return torch.tensor(baseline_np, dtype=torch.float32, device=tensor_x.device)[cite: 10]

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module
    ) -> List[List[float]]:
        """
        Computes position-specific importance metrics for an individual model instance[cite: 10].

        :param tensor_x: Pre-allocated encoded sequence tensor matrix shape (1, L, 4)[cite: 10].
        :type tensor_x: torch.Tensor
        :param model_instance: Initialized and synchronized PyTorch network object[cite: 10].
        :type model_instance: torch.nn.Module
        :return: Detached plain CPU-mapped list structure tracking gradient indices[cite: 5, 10].
        :rtype: List[List[float]]
        """
        if tensor_x.shape[0] > 1:
            raise ValueError("IntegratedGradientsStrategy only supports single-sequence operations.")[cite: 10]

        model_instance.eval()[cite: 9]
        model_instance.zero_grad()[cite: 10]
        
        baseline = self._generate_gc_matched_baseline(tensor_x)[cite: 10]
        total_gradients = torch.zeros_like(tensor_x)[cite: 10]

        # Path integral accumulation loop
        ## Step iteratively along the linear interpolation path between baseline and input[cite: 10]
        for step_idx in range(self.steps):
            alpha = float(step_idx) / float(self.steps)[cite: 10]
            interpolated_input = (baseline + alpha * (tensor_x - baseline)).detach()[cite: 10]
            interpolated_input.requires_grad_(True)[cite: 10]

            ### Execute model pass inside clean gradient computation tracking scopes[cite: 10]
            model_instance.zero_grad()[cite: 10]
            _, ratio = model_instance(interpolated_input)[cite: 9, 10]
            
            ### Compute scalar path integral reduction targets[cite: 10]
            backward_target = ratio.sum()[cite: 10]
            backward_target.backward()[cite: 10]

            ### Capture backpropagated graph parameters from the interpolation nodes[cite: 10]
            if interpolated_input.grad is not None:
                total_gradients += interpolated_input.grad[cite: 10]

        # Final average and detachment profiling
        ## Apply the fundamental theorem of calculus approximation to compute absolute attributions[cite: 10]
        avg_gradients = total_gradients / float(self.steps)[cite: 10]
        integrated_attributions = ((tensor_x - baseline) * avg_gradients).detach().abs()[cite: 10]
        
        ## Squeeze the batch dimension and serialize matrix coordinates to a list layout on CPU memory[cite: 5, 10]
        attributions_list = integrated_attributions[0].cpu().numpy().tolist()[cite: 5, 10]
        
        model_instance.zero_grad()[cite: 10]
        return attributions_list
    
    
class IntegratedGradientsStrategy(BaseAttributionStrategy):
    """
    Executes path integral Riemann integration on pre-allocated tensor spaces
    to track position-specific gradient attributions across targets.
    """

    def __init__(self, steps: int = 30, is_aggregated: bool = True):
        """
        Initializes the path integral approximation parameters.

        :param steps: The number of Riemann integration steps along the path trajectory.
        :type steps: int
        :param is_aggregated: Flag determining if the output schema aggregates multi-model responses.
        :type is_aggregated: bool
        """
        self.steps = steps
        self.is_aggregated = is_aggregated
        logger.info("IntegratedGradientsStrategy instantiated. Trajectory steps: %s", steps)

    def _generate_gc_matched_baseline(self, tensor_x: torch.Tensor) -> torch.Tensor:
        """
        Generates a uniform reference baseline matching the structural shape dimensions of the input.

        :param tensor_x: Input sequence tensor matrix of shape (B, L, 4).
        :type tensor_x: torch.Tensor
        :return: Reference matrix tracking uniform nucleotide probabilities on the same compute device.
        :rtype: torch.Tensor
        """
        batch_size, seq_len, channels = tensor_x.shape
        baseline_np = np.full((batch_size, seq_len, channels), 0.25, dtype=np.float32)
        return torch.tensor(baseline_np, dtype=torch.float32, device=tensor_x.device)

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.

        :param registered_models: List of available model identifiers inside the active manager.
        :type registered_models: List[str]
        :return: Map tracking output layout directives (Option B aggregation flag).
        :rtype: Dict[str, Any]
        """
        logger.debug("Resolving schema layout options for Integrated Gradients Strategy.")
        return {
            "is_aggregated": self.is_aggregated
        }

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module
    ) -> List[List[float]]:
        """
        Computes position-specific importance metrics for an already loaded tensor block.

        :param tensor_x: Pre-allocated encoded sequence tensor matrix shape (1, L, 4).
        :type tensor_x: torch.Tensor
        :param model_instance: Initialized and synchronized PyTorch network object.
        :type model_instance: torch.nn.Module
        :return: Detached plain CPU-mapped list structure tracking gradient indices.
        :rtype: List[List[float]]
        """
        if tensor_x.shape[0] > 1:
            logger.error("IntegratedGradientsStrategy received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("IntegratedGradientsStrategy only supports single-sequence operations.")

        # Allocation and baseline initialization
        ## Set models to evaluation tracks and clear graph residuals
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
            
            ### Compute scalar path integral reduction targets across structural objective coordinates
            backward_target = ratio.sum()
            backward_target.backward()

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
        logger.debug("Successfully serialized position importance tracking metrics matrix layout.")
        return attributions_list