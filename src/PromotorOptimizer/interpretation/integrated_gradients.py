# interpretation/integrated_gradients.py

import torch

from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one


class IntegratedGradientsInterpreter(BaseInterpreter):

    def __init__(self, steps: int = 50):
        self.steps = steps

    def explain(
        self,
        model_manager,
        sequence,
        model_type="ensemble"
    ):

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

            baseline = torch.zeros_like(x)

            total_gradients = torch.zeros_like(x)

            for alpha in torch.linspace(
                0,
                1,
                self.steps,
                device=device
            ):

                interp = (
                    baseline +
                    alpha * (x - baseline)
                ).detach()

                interp.requires_grad_(True)

                model.zero_grad()

                _, ratio = model(interp)

                score = ratio.mean()

                score.backward()

                total_gradients += interp.grad

            avg_grad = total_gradients / self.steps

            integrated_gradients = (
                (x - baseline) *
                avg_grad
            ).abs().squeeze(0)

            per_model_maps.append(
                integrated_gradients
            )

            with torch.no_grad():
                _, ratio = model(x)

            model_scores[name] = float(
                ratio.mean().cpu()
            )

        per_model_maps = torch.stack(
            per_model_maps
        )

        if model_type == "ensemble":
            importance = per_model_maps.mean(
                dim=0
            )
        else:
            importance = per_model_maps[0]

        return InterpretationResult(
            method_name="IntegratedGradients",
            importance_scores=importance.cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )