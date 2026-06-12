# Heading 1 (In Silico Mutagenesis Attribution Strategy Implementation)
## Operational mutation scanning backends, tensor modifications, and batch inference tools
import torch
from typing import Dict, List, Any

from .strategies.base import BaseAttributionStrategy
from ..utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class InSilicoMutagenesis(BaseAttributionStrategy):
    """
    Computes position-specific importance metrics by systematically substituting 
    every nucleotide position in the tensor space and measuring the absolute delta 
    in model prediction outputs[cite: 15].
    """

    def __init__(self, is_aggregated: bool = False, chunk_size: int = 1024):
        """
        Initializes the In Silico Mutagenesis interpretation strategy.

        :param is_aggregated: Flag determining if the strategy aggregates multi-model responses[cite: 1].
        :type is_aggregated: bool
        :param chunk_size: Sub-batch evaluation chunk size to prevent VRAM out-of-memory anomalies[cite: 15].
        :type chunk_size: int
        """
        self.is_aggregated = is_aggregated
        self.chunk_size = chunk_size
        logger.info("InSilicoMutagenesis strategy instantiated. Chunk evaluation footprint: %s", chunk_size)

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles[cite: 1].

        :param registered_models: List of available model identifiers inside the active manager[cite: 9].
        :type registered_models: List[str]
        :return: Map tracking output layout directives (is_aggregated flag status)[cite: 1].
        :rtype: Dict[str, Any]
        """
        logger.debug("Resolving schema layout options for In Silico Mutagenesis Strategy.")
        return {
            "is_aggregated": self.is_aggregated
        }

    def compute_tensor_attribution(
        self, 
        tensor_x: torch.Tensor, 
        model_instance: torch.nn.Module
    ) -> List[List[float]]:
        """
        Computes position-specific mutagenesis importance metrics for an already loaded tensor block[cite: 15].

        :param tensor_x: Pre-allocated encoded sequence tensor matrix shape (1, L, 4)[cite: 15].
        :type tensor_x: torch.Tensor
        :param model_instance: Initialized and synchronized PyTorch network object[cite: 15].
        :type model_instance: torch.nn.Module
        :return: Detached plain CPU-mapped list structure tracking prediction differences[cite: 15].
        :rtype: List[List[float]]
        """
        if tensor_x.shape[0] > 1:
            logger.error("InSilicoMutagenesis received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("InSilicoMutagenesis only supports single-sequence operations.")

        # Computational graph synchronization
        ## Ensure model tracking parameters are set to evaluation mode
        model_instance.eval()[cite: 15]
        model_instance.zero_grad()

        device = tensor_x.device
        seq_len = tensor_x.shape[1]

        # Establish reference baseline activity
        ## Execute a single forward pass on the wild-type tensor to derive the target baseline score[cite: 15]
        with torch.no_grad():
            _, base_ratio = model_instance(tensor_x)[cite: 15]
            base_score = base_ratio.mean().item()[cite: 15]

        # Combinatorial mutation tensor space instantiation
        ## Pre-allocate a dense matrix block containing all possible single-point nucleotide mutations[cite: 15]
        total_mutations = seq_len * 4
        x_mut_pool = tensor_x.repeat(total_mutations, 1, 1)

        ## Systematically re-map the multi-channel one-hot indices across the entire mutation matrix pool
        for pos in range(seq_len):
            for base_idx in range(4):
                row_idx = pos * 4 + base_idx
                ### Overwrite the channel dimension to reflect the target substitution mutation
                x_mut_pool[row_idx, pos, :] = 0.0
                x_mut_pool[row_idx, pos, base_idx] = 1.0

        # High-throughput sub-batched chunk streaming loop
        ## Process the mutated sequence array in discrete chunks to preserve memory integrity boundaries[cite: 15]
        ratio_mut_list = []
        for chunk_idx in range(0, total_mutations, self.chunk_size):
            x_chunk = x_mut_pool[chunk_idx : chunk_idx + self.chunk_size]
            
            with torch.no_grad():
                _, ratio_chunk = model_instance(x_chunk)[cite: 15]
                
            ### Cast chunk outputs to unified 1D vectors and pass them to host memory[cite: 15]
            if ratio_chunk.ndim == 0:
                ratio_chunk = ratio_chunk.unsqueeze(0)
            elif ratio_chunk.ndim > 1:
                ratio_chunk = ratio_chunk.mean(dim=[i for i in range(1, ratio_chunk.ndim)])
                
            ratio_mut_list.append(ratio_chunk.cpu())[cite: 15]

        ## Concatenate chunk outputs into a single aligned prediction tracking array[cite: 15]
        ratio_mut = torch.cat(ratio_mut_list, dim=0)[cite: 15]

        # Importance score redistribution matrix profiling
        ## Initialize the importance matrix and map absolute prediction differences back to spatial coordinates[cite: 15]
        importance_matrix = torch.zeros(seq_len, 4, device=device)[cite: 15]
        
        for pos in range(seq_len):
            for base_idx in range(4):
                row_idx = pos * 4 + base_idx
                mut_score = ratio_mut[row_idx].item()[cite: 15]
                
                ### Calculate absolute deviations from the unmutated baseline reference profile[cite: 15]
                importance_matrix[pos, base_idx] = abs(mut_score - base_score)[cite: 15]

        ## Convert the completed positional attribution tensor directly to a standard host list layout[cite: 15]
        attributions_list = importance_matrix.cpu().numpy().tolist()[cite: 15]

        model_instance.zero_grad()
        logger.debug("Successfully completed In Silico Mutagenesis matrix calculation pass.")
        return attributions_list