# Heading 1 (In Silico Mutagenesis Attribution Strategy Implementation)
## Operational mutation scanning backends, tensor modifications, and batch inference tools
import torch
from typing import Dict, List, Any

from .base_attribution_strategy import BaseAttributionStrategy
from ...loss_functions.strategies.base_loss_function import BaseLossObjective
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


class InSilicoMutagenesis(BaseAttributionStrategy):
    """
    Computes position-specific importance metrics by systematically measuring 
    the absolute delta in objective loss spaces caused by nucleotide substitutions.
    """  

    def __init__(self, objective: BaseLossObjective, chunk_size: int = 1024):
        """
        Initializes the discrete In Silico Mutagenesis interpretation strategy.

        :param objective: Injected cost function managing loss valuations.
        :type objective: BaseLossObjective
        :param chunk_size: Sub-batch evaluation chunk size to prevent VRAM out-of-memory anomalies.
        :type chunk_size: int
        """
        super().__init__(objective)
        self.chunk_size = chunk_size
        logger.info("Discrete InSilicoMutagenesis strategy instantiated. Chunk footprint: %s", chunk_size)

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.
        """
        logger.debug("Resolving discrete schema layout options for In Silico Mutagenesis Strategy.")
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
        Computes position-specific mutagenesis importance metrics in objective loss space.
        """
        if tensor_x.shape[0] > 1:
            logger.error("Discrete InSilicoMutagenesis received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("InSilicoMutagenesis only supports single-sequence operations.")

        # Computational graph synchronization
        model_instance.eval()
        model_instance.zero_grad()

        device = tensor_x.device
        seq_len = tensor_x.shape[1]

        # Establish reference baseline activity
        ## Execute a single forward pass on the wild-type tensor to derive the target baseline loss profile
        with torch.no_grad():
            _, base_ratio = model_instance(tensor_x)
            base_loss = self.objective.evaluate_tensor_loss(predictions=base_ratio, tensor_x=tensor_x, metadata=metadata).item()

        # Combinatorial mutation tensor space instantiation
        ## Pre-allocate a dense matrix block containing all possible single-point nucleotide mutations
        total_mutations = seq_len * 4
        x_mut_pool = tensor_x.repeat(total_mutations, 1, 1)

        for pos in range(seq_len):
            for base_idx in range(4):
                row_idx = pos * 4 + base_idx
                x_mut_pool[row_idx, pos, :] = 0.0
                x_mut_pool[row_idx, pos, base_idx] = 1.0

        # High-throughput sub-batched chunk streaming loop
        ## Process the mutated sequence array in discrete chunks to preserve memory integrity boundaries
        ratio_mut_list = []
        for chunk_idx in range(0, total_mutations, self.chunk_size):
            x_chunk = x_mut_pool[chunk_idx : chunk_idx + self.chunk_size]
            
            with torch.no_grad():
                _, ratio_chunk = model_instance(x_chunk)
                
            if ratio_chunk.ndim == 0:
                ratio_chunk = ratio_chunk.unsqueeze(0)
            elif ratio_chunk.ndim > 1:
                ratio_chunk = ratio_chunk.mean(dim=[i for i in range(1, ratio_chunk.ndim)])
                
            ratio_mut_list.append(ratio_chunk.cpu())

        ratio_mut = torch.cat(ratio_mut_list, dim=0)

        # Importance score redistribution matrix profiling
        ## Initialize the importance matrix and map absolute loss differences back to spatial coordinates
        importance_matrix = torch.zeros(seq_len, 4, device=device)

        for pos in range(seq_len):
            for base_idx in range(4):
                row_idx = pos * 4 + base_idx
                mut_pred = ratio_mut[row_idx].unsqueeze(0).to(device)
                
                ### Differentiable objective transformation without explicit method checking
                mut_loss = self.objective.evaluate_tensor_loss(predictions=mut_pred, tensor_x=tensor_x, metadata=metadata).item()
                importance_matrix[pos, base_idx] = abs(mut_loss - base_loss)

        ## Convert the completed positional attribution tensor directly to a standard host list layout
        attributions_list = importance_matrix.cpu().numpy().tolist()

        model_instance.zero_grad()
        logger.debug("Successfully completed In Silico Mutagenesis matrix calculation pass.")
        return attributions_list


class InSilicoMutagenesisAggregated(BaseAttributionStrategy):
    """
    Computes position-specific importance metrics by systematically substituting 
    every nucleotide position in the tensor space and measuring the absolute delta 
    in model prediction outputs for ensemble tracking targets.
    """

    def __init__(self, objective: BaseLossObjective, chunk_size: int = 1024):
        """
        Initializes the In Silico Mutagenesis interpretation strategy.

        :param objective: Injected cost function managing loss valuations.
        :type objective: BaseLossObjective
        :param chunk_size: Sub-batch evaluation chunk size to prevent VRAM out-of-memory anomalies.
        :type chunk_size: int
        """
        super().__init__(objective)
        self.chunk_size = chunk_size
        logger.info("InSilicoMutagenesisAggregated strategy instantiated. Chunk footprint: %s", chunk_size)

    def resolve_output_schema(self, registered_models: List[str]) -> Dict[str, Any]:
        """
        Determines the output configuration dictionary structural mapping key profiles.
        """
        logger.debug("Resolving schema layout options for In Silico Mutagenesis Aggregated Strategy.")
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
        Computes position-specific mutagenesis importance metrics for an already loaded ensemble tensor block.
        """
        if tensor_x.shape[0] > 1:
            logger.error("InSilicoMutagenesisAggregated received high-volume batch slice: %s", tensor_x.shape[0])
            raise ValueError("InSilicoMutagenesisAggregated only supports single-sequence operations.")

        model_instance.eval()
        model_instance.zero_grad()

        device = tensor_x.device
        seq_len = tensor_x.shape[1]

        with torch.no_grad():
            _, base_ratio = model_instance(tensor_x)
            base_loss = self.objective.evaluate_tensor_loss(predictions=base_ratio, tensor_x=tensor_x, metadata=metadata).item()

        total_mutations = seq_len * 4
        x_mut_pool = tensor_x.repeat(total_mutations, 1, 1)

        for pos in range(seq_len):
            for base_idx in range(4):
                row_idx = pos * 4 + base_idx
                x_mut_pool[row_idx, pos, :] = 0.0
                x_mut_pool[row_idx, pos, base_idx] = 1.0

        ratio_mut_list = []
        for chunk_idx in range(0, total_mutations, self.chunk_size):
            x_chunk = x_mut_pool[chunk_idx : chunk_idx + self.chunk_size]
            
            with torch.no_grad():
                _, ratio_chunk = model_instance(x_chunk)
                
            if ratio_chunk.ndim == 0:
                ratio_chunk = ratio_chunk.unsqueeze(0)
            elif ratio_chunk.ndim > 1:
                ratio_chunk = ratio_chunk.mean(dim=[i for i in range(1, ratio_chunk.ndim)])
                
            ratio_mut_list.append(ratio_chunk.cpu())

        ratio_mut = torch.cat(ratio_mut_list, dim=0)

        importance_matrix = torch.zeros(seq_len, 4, device=device)

        for pos in range(seq_len):
            for base_idx in range(4):
                row_idx = pos * 4 + base_idx
                mut_pred = ratio_mut[row_idx].unsqueeze(0).to(device)
                
                mut_loss = self.objective.evaluate_tensor_loss(predictions=mut_pred, tensor_x=tensor_x, metadata=metadata).item()
                importance_matrix[pos, base_idx] = abs(mut_loss - base_loss)

        attributions_list = importance_matrix.cpu().numpy().tolist()
        model_instance.zero_grad()
        return attributions_list