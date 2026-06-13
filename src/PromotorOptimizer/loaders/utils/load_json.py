# Heading 1 (Modernized Log Ingestion Space)
## High-performance multi-layer JSON trajectory extraction utilities
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from ..add_metrics.orchestror import calculate_trajectory_metrics
from ...utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)


def parse_json_folder(folder_path: Path | str) -> pd.DataFrame:
    """
    Parses a directory containing new schema JSON optimization logs and flattens
    them directly into a long-format analytical DataFrame.

    :param folder_path: Path to the target log file directory.
    :type folder_path: Path | str
    :return: Combined long-format DataFrame with extracted structural metadata.
    :rtype: pd.DataFrame
    """
    folder = Path(folder_path)
    all_steps = []

    if not folder.exists():
        logger.error("Target analysis log directory does not exist: %s", folder)
        raise FileNotFoundError(f"Directory not found: {folder}")

    # File discovery loop
    for filepath in folder.glob("*.json"):
        file_id = filepath.name
        logger.debug("Ingesting modern trajectory dataset file: %s", file_id)

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error("Skipping corrupted or empty JSON frame target: %s", file_id)
                continue

        # Hierarchy parsing: sequence_id -> optimizers -> interpreters -> models -> steps
        for seq_id, seq_payload in data.items():
            optimizers_dict = seq_payload.get("optimizers", {})

            for optimizer_name, optimizer_payload in optimizers_dict.items():
                interpreters_dict = optimizer_payload.get("interpreters", {})

                for interpreter_name, interpreter_payload in interpreters_dict.items():
                    models_dict = interpreter_payload.get("models", {})

                    for model_key, model_payload in models_dict.items():
                        steps = model_payload.get("steps", [])
                        
                        # Set analytical taxonomy attributes
                        ## Determine operational mode indicators
                        combined_context = f"{file_id}_{seq_id}_{optimizer_name}".lower()
                        job_type = "reconstruction" if "reconstruction" in combined_context else "optimization"
                        
                        ## Extract model architecture sub-type labels
                        model_type = "aggregated" if model_key == "aggregated_models" else model_key

                        for step in steps:
                            current_seq = step.get("current_sequence", "")
                            if not current_seq:
                                continue

                            seq_len = len(current_seq)
                            gc_count = current_seq.upper().count("G") + current_seq.upper().count("C")
                            gc_content = gc_count / seq_len if seq_len > 0 else 0.0

                            # Score resolution
                            ## Map cross-sectional target metrics cleanly based on layout definitions
                            predictions = step.get("models_predictions", {})
                            if model_key == "aggregated_models":
                                ensemble_score = float(np.mean(list(predictions.values()))) if predictions else 0.0
                            else:
                                ensemble_score = float(predictions.get(model_key, 0.0))

                            # Extract weights for the primary tracking path
                            attributions = step.get("models_attributions", {})
                            if model_key == "aggregated_models" and attributions:
                                first_model = list(attributions.keys())[0]
                                interpreter_weights = attributions[first_model]
                            else:
                                interpreter_weights = attributions.get(model_key, None)

                            row = {
                                "file_id": file_id,
                                "sequence_name": seq_id,
                                "interpreter": interpreter_name,
                                "optimizer": optimizer_name,
                                "job_type": job_type,
                                "model_type": model_type,
                                "iteration": int(step.get("iteration", 0)),
                                "ensemble_score": ensemble_score,
                                "gc_content": gc_content,
                                "temperature": float(step.get("temperature", 0.0)),
                                "current_sequence": current_seq,
                                "current_beam_population": step.get("beam_population", []),
                                "interpreter_weights": interpreter_weights
                            }

                            # Flatten cross-model performance distributions dynamically
                            for m_name, m_score in predictions.items():
                                row[f"score_{m_name.replace(' ', '_')}"] = float(m_score)

                            all_steps.append(row)

    if not all_steps:
        logger.warning("Zero valid trajectory rows isolated across log folder files.")
        return pd.DataFrame()

    trajectory_df = pd.DataFrame(all_steps)
    
    # Enforce strict chronological row sorting
    trajectory_df.sort_values(
        by=["file_id", "sequence_name", "interpreter", "optimizer", "iteration"], 
        inplace=True, 
        ignore_index=True
    )
    return trajectory_df