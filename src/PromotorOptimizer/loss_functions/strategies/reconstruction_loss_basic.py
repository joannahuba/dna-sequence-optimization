import torch
import numpy as np
from typing import Dict, Any

from .base_loss_function import BaseLossObjective
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class ReconstructionLossObjective(BaseLossObjective):
    """
    Differentiable target expression profile restoration objective].

    For gradient backpropagation in interpreters, it evaluates the Mean Squared Error (MSE):
    $$L = \\frac{1}{M}\\sum_{m=1}^{M} (\\hat{y}_m - y_{\\text{target}})^2$$

    For selection fitness scoring on CPU, it evaluates the negative absolute deviation (higher is better):
    $$\\text{Fitness} = -|\\bar{\\hat{y}} - y_{\\text{target}}|$$
    """

    def evaluate_tensor_loss(
        self, 
        predictions: torch.Tensor, 
        tensor_x: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Computes differentiable MSE loss against the target expression constraint].
        """
        target_expression = metadata.get("target_value", 0.0)
        target_tensor = torch.tensor(target_expression, dtype=predictions.dtype, device=predictions.device)
        
        ## Compute mean square deviation to track distance from target expression profile]
        loss = torch.mean((predictions - target_tensor) ** 2)
        return loss

    def evaluate_numpy_fitness(
        self, 
        predictions: Dict[str, float], 
        sequence: str, 
        metadata: Dict[str, Any]
    ) -> float:
        """
        Computes negative absolute distance fitness scores for CPU candidate ranking].
        """
        target_expression = metadata.get("target_value", 0.0)
        scores = list(predictions.values())
        mean_score = sum(scores) / len(scores) if scores else 0.0
        
        ## Calculate absolute distance from targeting coordinates (negated for maximization)]
        fitness = -abs(mean_score - target_expression)
        return float(fitness)