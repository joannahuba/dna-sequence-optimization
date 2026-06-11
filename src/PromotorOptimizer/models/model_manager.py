from typing import Dict, List, Any
import numpy as np
import torch

from .base_model_manager import BaseModelManager
from ..utils.preprocessing import encode_batch
from ..utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


class ModelManager(BaseModelManager):

    def __init__(
        self,
        models_dict: Dict[str, Dict[str, Any]],
        batch_size: int = 256,
    ):
        self.models_dict = models_dict
        self.batch_size = batch_size

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # =====================================================
    # HELPERS
    # =====================================================

    def get_models(self):
        return self.models_dict

    def get_device(self):
        return self.device

    def get_model_names(self):
        return list(self.models_dict.keys())

    # =====================================================
    # TENSOR PREDICTION
    # =====================================================

    def predict_tensor(
        self,
        tensor: torch.Tensor
    ):

        outputs = {}

        tensor = tensor.to(self.device)

        for name, meta in self.models_dict.items():

            model = meta["model"]

            model.to(self.device)
            model.eval()

            with torch.no_grad():

                active, ratio = model(tensor)

            outputs[name] = {
                "active": active.detach().cpu(),
                "ratio": ratio.detach().cpu()
            }

        return outputs

    # =====================================================
    # RAW STRING PREDICTION
    # =====================================================

    def predict_sequences(
        self,
        sequences: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes high-throughput multi-model inference by globally pre-encoding 
        sequences and minimizing host-to-device context switches.

        :param sequences: Cleaned list of unique nucleotide strings.
        :type sequences: List[str]
        :return: Nested map tracking scores per sequence per model.
        :rtype: Dict[str, Dict[str, Any]]
        """
        # Global pre-encoding execution block
        ## Encode all sequence strings to eliminate iterative python execution overhead
        logger.debug("Starting pre-encoding for sequence array of volume: %s", len(sequences))
        X_all = encode_batch(sequences)
        
        results = {
            seq: {}
            for seq in sequences
        }

        # Multi-model prediction pipeline orchestration
        ## Outer Loop: Prepare each model exactly once to prevent CUDA context switching overhead
        for model_name, meta in self.models_dict.items():
            logger.debug("Synchronizing hardware runtime environment for model: %s", model_name)
            model = meta["model"]
            model.to(self.device)
            model.eval()

            ## High-throughput batch chunk streaming loop
            for i in range(0, len(sequences), self.batch_size):
                batch_seqs = sequences[i : i + self.batch_size]
                X_chunk = X_all[i : i + self.batch_size]
                
                ### Transfer matrix slices to compute device
                X = torch.tensor(
                    X_chunk,
                    dtype=torch.float32,
                    device=self.device
                )

                with torch.no_grad():
                    _, ratio = model(X)

                ### Fetch prediction arrays back to host memory space
                ratio_np = ratio.detach().cpu().numpy()

                ### Map evaluation scalars to persistent dictionary structures
                for idx, seq in enumerate(batch_seqs):
                    seq_output = ratio_np[idx]
                    
                    if seq_output.size == 1:
                        #### Handle standard single-task target output scaling
                        results[seq][model_name] = float(seq_output.item())
                    else:
                        #### Preserve complete raw array mapping profiles for multi-task outputs
                        results[seq][model_name] = seq_output.tolist()

        logger.debug("Completed parallel sequence validation prediction mapping successfully.")
        return results

    # =====================================================
    # SINGLE SEQUENCE SCORE
    # =====================================================

    def score_sequence(
        self,
        sequence: str,
        penalty_std: float = 0.2
    ):

        scores = self.predict_sequences(
            [sequence]
        )[sequence]

        scores = np.array(
            list(scores.values())
        )

        mean = scores.mean()
        std = scores.std()

        return {
            "mean": float(mean),
            "std": float(std),
            "fitness": float(
                mean -
                penalty_std * std
            )
        }

    # =====================================================
    # MULTI SEQUENCE SCORE
    # =====================================================

    # @maxi7524 #TODO - sprawdzić czy się to pojawia w uczeniu, w jaki sposób 
    def score_sequences(
        self,
        sequences: List[str],
        penalty_std: float = 0.2
    ):

        raw = self.predict_sequences(
            sequences
        )

        output = {}

        for seq, model_scores in raw.items():

            scores = np.array(
                list(model_scores.values())
            )

            mean = scores.mean()
            std = scores.std()

            output[seq] = {
                "mean": float(mean),
                "std": float(std),
                "fitness": float(
                    mean -
                    penalty_std * std
                )
            }

        return output

    # =====================================================
    # ENSEMBLE STATS FROM TENSOR
    # =====================================================

    def ensemble_predict(
        self,
        tensor: torch.Tensor
    ):

        outputs = self.predict_tensor(
            tensor
        )

        scores = []

        for result in outputs.values():

            scores.append(
                result["ratio"]
                .mean()
                .item()
            )

        scores = np.array(scores)

        return {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max())
        }