import os
import glob
import torch
import logging
from typing import List, Dict, Any

from .extractor import extract_unique_sequences
from .evaluator import CrossModelEvaluator
from .updater import update_json_with_predictions
from ...models.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Master pipeline control block
## Coordinate cross-model batch inference validation across directories
def run_cross_validation_pipeline(
    json_dir: str,
    output_dir: str,
    model_names: List[str],
    device: str = "cuda",
    batch_size: int = 64
) -> None:
    """
    Executes the master cross-validation pipeline over all optimization log files.

    :param json_dir: Path to the directory containing input JSON optimization logs.
    :type json_dir: str
    :param output_dir: Path to the directory where updated JSON logs will be saved.
    :type output_dir: str
    :param model_names: List of model identification keys to request from ModelRegistry.
    :type model_names: List[str]
    :param device: Compute device target for parallel matrix inference. Default is 'cuda'.
    :type device: str
    :param batch_size: Maximum volume of sequences processed concurrently. Default is 64.
    :type batch_size: int
    :return: None
    :rtype: None
    """
    # Environment initialization
    ## Verify output directory paths
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Infrastructure fetching
    ## Resolve models utilizing the dynamic central ModelRegistry
    model_registry_data = ModelRegistry.load(model_names)
    unwrapped_models = {k: v["model"] for k, v in model_registry_data.items()}

    ## Initialize the unified evaluation engine
    evaluator = CrossModelEvaluator(model_dict=unwrapped_models, device=device)

    ## Scan directory path for target log files
    json_pattern = os.path.join(json_dir, "*.json")
    target_files = glob.glob(json_pattern)

    # Document batch execution loop
    for file_path in target_files:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(output_dir, file_name)
        
        logger.info(f"[PIPELINE] Starting cross-validation for target: {file_name}")

        try:
            ### Extract unique sequences using the corrected configuration dictionary keys
            raw_data, unique_seqs = extract_unique_sequences(file_path)
            
            if not unique_seqs:
                logger.warning(f"[PIPELINE] No valid sequences isolated in file: {file_name}")
                continue

            logger.info(f"[PIPELINE] Isolated {len(unique_seqs)} unique variants for parallel GPU execution")

            ### Compute parallel multi-model matrix predictions on GPU
            predictions_map = evaluator.compute_predictions(
                sequences=unique_seqs, 
                batch_size=batch_size
            )

            ### Inject validation metrics scores back into the structural dump
            update_json_with_predictions(
                json_path=file_path,
                output_path=destination_path,
                predictions=predictions_map
            )
            
            logger.info(f"[PIPELINE] Successfully exported enriched dataset log to: {destination_path}")

        except Exception as error_exception:
            logger.error(f"[PIPELINE] Critical execution failure on target file {file_name}: {str(error_exception)}")