# interpretation/saliency.py
import torch
from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


class SaliencyInterpreter(BaseInterpreter):

    def explain(
        self,
        model_manager,
        sequence: str,
        model_type="ensemble"
    ):

        logger.info(f"[Saliency] Start | model_type={model_type} | seq_len={len(sequence)}")

        device = model_manager.get_device()
        models = model_manager.get_models()

        x = torch.tensor(
            encode_one(sequence),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)  # (1, L, 4)

        per_model_maps = []
        model_scores = {}

        for name, meta in models.items():

            model = meta["model"]
            model.to(device)
            model.eval()

            x_req = x.clone().detach().requires_grad_(True)

            _, ratio = model(x_req)
            score = ratio.mean()

            model_scores[name] = float(score.detach().cpu())
            logger.info(f"[Saliency] model={name} score={model_scores[name]:.6f}")

            model.zero_grad()
            score.backward()

            grad = x_req.grad  # (1, L, 4)

            # IMPORTANT: keep per-base importance
            saliency = grad.abs().squeeze(0)  # (L, 4)

            logger.info(
                f"[Saliency] model={name} grad_stats="
                f"min={saliency.min().item():.4f} "
                f"max={saliency.max().item():.4f}"
            )

            per_model_maps.append(saliency)

        per_model_maps = torch.stack(per_model_maps)  # (M, L, 4)

        if model_type == "ensemble":
            importance = per_model_maps.mean(dim=0)
            logger.info("[Saliency] Using ENSEMBLE averaging")
        else:
            importance = per_model_maps[0]
            logger.info("[Saliency] Using SINGLE model mode")

        logger.info("[Saliency] Finished")


        return InterpretationResult(
            method_name="Saliency",
            importance_scores=importance.detach().cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )