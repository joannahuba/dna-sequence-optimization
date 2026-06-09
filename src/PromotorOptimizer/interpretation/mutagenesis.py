import torch
import logging
from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one

logger = logging.getLogger(__name__)

BASES = ["A", "C", "G", "T"]


class InSilicoMutagenesis(BaseInterpreter):

    def explain(self, model_manager, sequence, model_type="ensemble"):

        logger.info(f"[Mutagenesis] Start | seq_len={len(sequence)}")

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

            logger.info(f"[Mutagenesis] model={name} base_score={base_score:.6f}")

            importance = torch.zeros(seq_len, 4, device=device)

            for pos in range(seq_len):

                if pos % 20 == 0:
                    logger.info(f"[Mutagenesis] model={name} position={pos}/{seq_len}")

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

                    importance[pos, base_idx] = abs(score - base_score)

            importance_maps.append(importance)

        importance_maps = torch.stack(importance_maps)

        importance = importance_maps.mean(dim=0) if model_type == "ensemble" else importance_maps[0]

        logger.info("[Mutagenesis] Finished")

        return InterpretationResult(
            method_name="Mutagenesis",
            importance_scores=importance.cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )