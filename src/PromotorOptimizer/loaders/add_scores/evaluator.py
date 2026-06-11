import torch
import numpy as np
from typing import List, Dict
 


class CrossModelEvaluator:
    """
    Coordinates multi-model batch inference on unique biological sequences.
    This class handles the vectorization, one-hot encoding, and GPU-accelerated
    forward-pass execution for Deepstar, Second Deepstarr, and Model Original.

    :param model_dict: Dictionary mapping model names to their loaded PyTorch instances.
    :type model_dict: dict
    :param device: Target compute device for tensor operations.
    :type device: str
    """

    def __init__(self, model_dict: dict, device: str = "cuda"):
        self.device = device
        self.models = model_dict

        # Model environment synchronization
        ## Set all loaded network architectures to evaluation mode
        for model in self.models.values():
            model.to(self.device)
            model.eval()

    def _encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """
        Transforms a list of nucleotide strings into a single dense 3D tensor.

        :param sequences: List of DNA sequence strings to vectorize.
        :type sequences: list of str
        :return: One-hot encoded tensor of shape (B, L, 4).
        :rtype: torch.Tensor
        """
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        batch_encoded = []

        # Sequence tensor packing loop
        for seq in sequences:
            seq_upper = seq.upper()
            encoded = np.zeros((len(seq_upper), 4), dtype=np.float32)
            
            for idx, base in enumerate(seq_upper):
                if base in mapping:
                    encoded[idx, mapping[base]] = 1.0
            
            batch_encoded.append(encoded)

        return torch.tensor(np.array(batch_encoded), dtype=torch.float32, device=self.device)

    def compute_predictions(self, sequences: List[str], batch_size: int = 64) -> Dict[str, Dict[str, float]]:
        """
        Executes parallelized batch inference across all registered models.

        :param sequences: Cleaned list of unique nucleotide strings.
        :type sequences: list of str
        :param batch_size: Maximum volume of sequences processed concurrently. Default is 64.
        :type batch_size: int
        :return: Nested map tracking scores per sequence per model: {seq: {model_name: score}}.
        :rtype: dict
        """
        results: Dict[str, Dict[str, float]] = {seq: {} for seq in sequences}

        # Batch partitioning and processing orchestration
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            
            ## Construct structural input tensors for the current batch
            x_input = self._encode_batch(batch_seqs)

            ## Execute forward pass loops across model configurations
            for model_name, model_obj in self.models.items():
                with torch.no_grad():
                    ### Perform model inference to calculate expression outputs
                    _, ratio = model_obj(x_input)
                    
                    ### Reduce matrix dimensions to capture mean scalar scores
                    scores = ratio.mean(dim=[i for i in range(1, ratio.ndim)]).cpu().numpy()

                ### Map calculated predictions back to individual sequence tracking keys
                for idx, seq in enumerate(batch_seqs):
                    results[seq][model_name] = float(scores[idx])

        return results