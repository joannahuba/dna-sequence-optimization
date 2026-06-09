# interpretation/mutagenesis.py

import torch
from .base import BaseInterpreter

import torch

from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one


BASES = ["A", "C", "G", "T"]


class InSilicoMutagenesis(BaseInterpreter):

    def explain(
        self,
        model_manager,
        sequence,
        model_type="ensemble"
    ):

        device = model_manager.get_device()

        models = model_manager.get_models()

        seq_len = len(sequence)

        importance_maps = []

        model_scores = {}

        for name, meta in models.items():

            model = meta["model"]

            base_tensor = torch.tensor(
                encode_one(sequence),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():

                _, ratio = model(base_tensor)

                base_score = ratio.mean().item()

            model_scores[name] = base_score

            importance = torch.zeros(
                seq_len,
                4,
                device=device
            )

            for pos in range(seq_len):

                current_base = sequence[pos]

                for base_idx, new_base in enumerate(BASES):

                    if new_base == current_base:
                        continue

                    mutated = list(sequence)

                    mutated[pos] = new_base

                    mutated = "".join(mutated)

                    x = torch.tensor(
                        encode_one(mutated),
                        dtype=torch.float32,
                        device=device
                    ).unsqueeze(0)

                    with torch.no_grad():

                        _, ratio = model(x)

                        score = ratio.mean().item()

                    importance[pos, base_idx] = abs(
                        score - base_score
                    )

            importance_maps.append(
                importance
            )

        importance_maps = torch.stack(
            importance_maps
        )

        if model_type == "ensemble":
            importance = importance_maps.mean(
                dim=0
            )
        else:
            importance = importance_maps[0]

        return InterpretationResult(
            method_name="Mutagenesis",
            importance_scores=importance.cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )