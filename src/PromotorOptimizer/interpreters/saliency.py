# interpretation/saliency.py
import torch
import numpy as np
from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one
import logging

logger = logging.getLogger(__name__)


class SaliencyInterpreter(BaseInterpreter):

    def explain(self, model_manager, sequence: str, model_type="ensemble"):
        # Zachowujemy oryginalną metodę dla kompatybilności wstecznej
        return self.explain_batch(model_manager, [sequence], model_type)[0]

    def explain_batch(self, model_manager, sequences: list, model_type="ensemble"):
        """
        Computes saliency attribution maps for a batch of sequences concurrently on GPU.

        :param model_manager: Unified evaluation suite manager coordination stack.
        :param sequences: List of DNA sequence strings currently in the beam pool.
        :type sequences: list of str
        :param model_type: Averaging methodology flag ('ensemble' or 'single').
        :type model_type: str
        :return: List of InterpretationResult instances matching the input batch order.
        :rtype: list
        """
        device = model_manager.get_device()
        models = model_manager.get_models()
        batch_size = len(sequences)

        # Vectorized input encoding
        ## Stack all encoded sequences into a single dense matrix component tensor
        encoded_list = np.array([encode_one(seq) for seq in sequences])
        x = torch.tensor(encoded_list, dtype=torch.float32, device=device) # Shape: (B, L, 4)

        per_model_maps = []
        model_scores = {name: [] for name in models.keys()}

        for name, meta in models.items():
            model = meta["model"]
            #TODO check if it was not instantiated earlier
            # model.to(device)
            model.eval()

            # Enable gradient capture across the entire spatial sequence batch
            x_req = x.clone().detach().requires_grad_(True)

            _, ratio = model(x_req) # Predict expression profile values
            score = ratio.mean()   # Dynamic continuous reduction metric

            model.zero_grad()
            score.backward()

            # Extract instantaneous derivatives across all elements parallelly
            grad = x_req.grad # Shape: (B, L, 4)
            saliency = grad.abs() # Shape: (B, L, 4)
            per_model_maps.append(saliency)

            # Dimensionality-agnostic evaluation score parsing
            with torch.no_grad():
                _, out_ratio = model(x)
                for b_idx in range(batch_size):
                    ### Safely extract scalar metric values across varying tensor shapes
                    if out_ratio.ndim == 0:
                        score_val = float(out_ratio.item())
                    elif out_ratio.ndim == 1:
                        score_val = float(out_ratio[b_idx].item())
                    else:
                        score_val = float(out_ratio[b_idx].mean().item())
                    model_scores[name].append(score_val)

        per_model_maps = torch.stack(per_model_maps) # Shape: (M, B, L, 4)

        # Reduce gradients according to model coordination configuration
        if model_type == "ensemble":
            importance_batch = per_model_maps.mean(dim=0) # Shape: (B, L, 4)
        else:
            importance_batch = per_model_maps[0]

        # Compile and un-batch results array
        results = []
        for b_idx in range(batch_size):
            single_scores = {name: model_scores[name][b_idx] for name in models.keys()}
            results.append(
                InterpretationResult(
                    method_name="Saliency",
                    importance_scores=importance_batch[b_idx].cpu(),
                    sequence=sequences[b_idx],
                    model_scores=single_scores,
                    metadata={}
                )
            )
        return results