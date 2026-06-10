# interpretation/integrated_gradients.py
import torch
import logging
import numpy as np
from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one

logger = logging.getLogger(__name__)


class IntegratedGradientsInterpreter(BaseInterpreter):

    def __init__(self, steps: int = 30): # Domyślnie zmniejszamy kroki do 30 dla optymalizacji prędkości
        self.steps = steps

    def _generate_gc_matched_baseline_batch(self, sequences: list, device: torch.device) -> torch.Tensor:
        baseline_list = []
        for sequence in sequences:
            seq_len = len(sequence)
            seq_upper = sequence.upper()
            gc_count = seq_upper.count('G') + seq_upper.count('C')
            at_count = seq_upper.count('A') + seq_upper.count('T')
            total = gc_count + at_count if (gc_count + at_count) > 0 else seq_len
            
            gc_fraction = gc_count / total
            gc_deviation = np.random.uniform(-0.05, 0.05)
            target_gc = max(0.0, min(1.0, gc_fraction + gc_deviation))
            target_at = 1.0 - target_gc
            
            p_a = p_t = target_at / 2.0
            p_c = p_g = target_gc / 2.0
            
            baseline_np = np.zeros((seq_len, 4))
            baseline_np[:, 0] = p_a
            baseline_np[:, 1] = p_c
            baseline_np[:, 2] = p_g
            baseline_np[:, 3] = p_t
            baseline_list.append(baseline_np)
            
        return torch.tensor(np.array(baseline_list), dtype=torch.float32, device=device) # (B, L, 4)

    def explain(self, model_manager, sequence: str, model_type="ensemble"):
        return self.explain_batch(model_manager, [sequence], model_type)[0]

    def explain_batch(self, model_manager, sequences: list, model_type="ensemble"):
        """
        Executes vectorized batch-wise Riemann trajectory integration for deep feature attribution.
        """
        device = model_manager.get_device()
        models = model_manager.get_models()
        batch_size = len(sequences)

        encoded_list = np.array([encode_one(seq) for seq in sequences])
        x = torch.tensor(encoded_list, dtype=torch.float32, device=device) # (B, L, 4)

        per_model_maps = []
        model_scores = {name: [] for name in models.keys()}

        for name, meta in models.items():
            model = meta["model"]
            model.eval()

            logger.info(f"[IG] model={name} processing started")

            baseline = self._generate_gc_matched_baseline_batch(sequences, device) # (B, L, 4)
            total_gradients = torch.zeros_like(x)

            # Concurrent path integration loops over batch elements
            for i, alpha in enumerate(torch.linspace(0, 1, self.steps, device=device)):
                interp = (baseline + alpha * (x - baseline)).detach()
                interp.requires_grad_(True)

                model.zero_grad()
                _, ratio = model(interp)
                score = ratio.mean()
                score.backward()

                total_gradients += interp.grad

                if i % 10 == 0 or i == self.steps - 1:
                    logger.info(f"[IG] model={name} step={i}/{self.steps}")

            avg_grad = total_gradients / self.steps
            integrated = (x - baseline) * avg_grad
            integrated = integrated.abs() # Shape: (B, L, 4)
            per_model_maps.append(integrated)



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
                    
                    #### Print individual sample score from inside the processed batch tracking array
                    logger.info(f"[IG] model={name} sample={b_idx} score={score_val:.6f}")

        per_model_maps = torch.stack(per_model_maps) # (M, B, L, 4)
        importance_batch = per_model_maps.mean(dim=0) if model_type == "ensemble" else per_model_maps[0]

        logger.info("[IG] Finished")

        results = []
        for b_idx in range(batch_size):
            single_scores = {name: model_scores[name][b_idx] for name in models.keys()}
            results.append(
                InterpretationResult(
                    method_name="IntegratedGradients",
                    importance_scores=importance_batch[b_idx].cpu(),
                    sequence=sequences[b_idx],
                    model_scores=single_scores,
                    metadata={}
                )
            )
        return results