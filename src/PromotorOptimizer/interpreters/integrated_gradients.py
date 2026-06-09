import torch
import logging
from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one

logger = logging.getLogger(__name__)


class IntegratedGradientsInterpreter(BaseInterpreter):

    def __init__(self, steps: int = 50):
        self.steps = steps

    def explain(self, model_manager, sequence, model_type="ensemble"):

        logger.info(
            f"[IG] Start | steps={self.steps} | model_type={model_type}"
        )

        device = model_manager.get_device()
        models = model_manager.get_models()

        x = torch.tensor(
            encode_one(sequence),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        per_model_maps = []
        model_scores = {}

        for name, meta in models.items():

            model = meta["model"]
            model.eval()

            logger.info(f"[IG] model={name} processing started")

            baseline = torch.zeros_like(x)
            total_gradients = torch.zeros_like(x)

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
            integrated = integrated.abs().squeeze(0)

            per_model_maps.append(integrated)

            with torch.no_grad():
                _, ratio = model(x)

            model_scores[name] = float(ratio.mean().cpu())

            logger.info(
                f"[IG] model={name} final_score={model_scores[name]:.6f}"
            )

        per_model_maps = torch.stack(per_model_maps)

        importance = per_model_maps.mean(dim=0) if model_type == "ensemble" else per_model_maps[0]

        logger.info("[IG] Finished")

        return InterpretationResult(
            method_name="IntegratedGradients",
            importance_scores=importance.cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )