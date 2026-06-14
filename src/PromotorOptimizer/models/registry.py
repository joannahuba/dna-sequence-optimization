# models/registry.py

from typing import Dict, Any
import torch

from .loaders.deepstarr_loader import load_deepstarr
from .loaders.zero_loader import load_model_zero
from .loaders.deep_starr_second import load_deep_starr_second
from .loaders.original_modified import load_original_modified
from .datasets.dna_dataset import DNADataset
from .datasets.dna_no_adapters_dataset import DNADatasetNoAdapters


class ModelRegistry:

    @staticmethod
    def load(model_names: list[str]) -> Dict[str, Dict[str, Any]]:

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        registry = {}

        for name in model_names:

            # -------------------------
            # MODEL ZERO
            # -------------------------
            if name in ["model_zero", "zero_test_model"]:
                # TODO - add later some configurator of forward functions 
                # REMARK: it does not work because of different forward structure
                model = load_model_zero(
                    "data/checkpoints/zero_test_model.pth",
                    device
                )

                registry[name] = {
                    "model": model,
                    "dataset_class": DNADataset
                }

            elif name in ["original_modified", "noadapters_model"]:

                model = load_original_modified(
                    "data/checkpoints/original_modified.pth",
                    device
                )

                registry[name] = {
                    "model": model,
                    "dataset_class": DNADatasetNoAdapters
                }


            # -------------------------
            # DEEPSTARR
            # -------------------------
            elif name in ["deepstarr", "scheduler_plateau"]:

                model = load_deepstarr(
                    "data/checkpoints/scheduler_plateau.pth",
                    device
                )

                registry[name] = {
                    "model": model,
                    "dataset_class": DNADataset
                }
            
            elif name in ["deepstarr_second"]:

                model = load_deep_starr_second(
                    "data/checkpoints/starr_original.pth",
                    device
                )

                registry[name] = {
                    "model": model,
                    "dataset_class": DNADatasetNoAdapters
                }


            else:
                raise ValueError(f"Unknown model: {name}")

        return registry