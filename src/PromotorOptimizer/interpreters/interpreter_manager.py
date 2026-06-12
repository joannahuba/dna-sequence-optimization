from typing import List, Dict, Any
import torch

from .strategies.base_attribution_strategy import AttributionStrategy
from ..models.model_manager import ModelManager
from ..utils.preprocessing import encode_batch
from utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class InterpreterManager:
    """
    Orchestrates target sequence attribution evaluation profiles for a single 
    deep learning model utilizing isolated gradient tracking strategies.
    """

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.get_device()

    def interpret_single_model_step(
        self, 
        sequence: str, 
        model_name: str,
        strategy: AttributionStrategy
    ) -> List[List[float]]:
        """
        Computes raw gradients for a single sequence state against one specific model.

        :param sequence: The exact DNA sequence string at the current optimization step.
        :type sequence: str
        :param model_name: Identifier key of the specific model to target.
        :type model_name: str
        :param strategy: Instantiated strategy object defining the gradient calculation mode.
        :type strategy: AttributionStrategy
        :return: Explicit non-aggregated attribution matrix list for the requested model.
        :rtype: List[List[float]]
        """
        # Vectorized encoding phase
        ## Parse text input strings into numeric floating point arrays
        X_raw = encode_batch([sequence])
        
        models_dict = self.model_manager.get_models()
        if model_name not in models_dict:
            logger.error("Requested model key validation failed against registry: %s", model_name)
            raise KeyError(f"Model {model_name} not found in ModelManager registry stack.")

        meta = models_dict[model_name]
        model = meta["model"]
        
        # Computing framework alignment
        ## Move network components to the active hardware compute device
        model.to(self.device)
        model.eval()

        X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=self.device)

        try:
            ### Delegate attribution extraction to the current strategy instance
            attribution_tensor = strategy.compute_attribution(model, X_tensor)
            attribution_np = attribution_tensor.cpu().numpy()
            
            ### Yield raw nested sequence matrix configuration layout
            return attribution_np[0].tolist()
            
        except Exception as error_exception:
            logger.error("Isolated target attribution sequence mapping failed for model context: %s", model_name, exc_info=True)
            raise error_exception