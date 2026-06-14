import os
import glob
from typing import List

from .extractor import extract_unique_sequences
from .updater import update_json_with_predictions
from ...models.registry import ModelRegistry
from ...models.model_manager import ModelManager
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def run_cross_validation_pipeline(
    json_dir: str,
    output_dir: str,
    model_names: List[str],
    device: str = "cuda",
    # REMARK
    # It was tested in `resources/tutorial_notebooks/tutorial.ipynb`, for gtx 1060 with 6G vRAM it it optimal
    batch_size: int = 64
) -> None:
    """
    Executes the master cross-validation pipeline using optimized ModelManager batching streams.

    :param json_dir: Path to the directory containing input JSON optimization logs.
    :type json_dir: str
    :param output_dir: Path to the directory where updated JSON logs will be saved.
    :type output_dir: str
    :param model_names: List of model identification keys to request from ModelRegistry.
    :type model_names: List[str]
    :param device: Compute device target for parallel matrix inference. Default is 'cuda'.
    :type device: str
    :param batch_size: Maximum volume of sequences processed concurrently. Default is 1024.
    :type batch_size: int
    :return: None
    :rtype: None
    :raises FileNotFoundError: If the input directory does not exist or contains no JSON files.
    """
    # Environment initialization
    ## Verify input directory presence
    if not os.path.exists(json_dir):
        logger.error("Target execution directory verification failed: %s", json_dir)
        raise FileNotFoundError(f"[PIPELINE] Target input directory does not exist: {json_dir}")

    ## Verify output directory paths
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## Scan directory path for target log files
    json_pattern = os.path.join(json_dir, "*.json")
    target_files = glob.glob(json_pattern)

    if not target_files:
        logger.error("Empty directory matched for global configuration path pattern: %s", json_pattern)
        raise FileNotFoundError(f"[PIPELINE] No target JSON files discovered matching pattern: {json_pattern}")

    # Shared infrastructure resource pooling
    ## Resolve structural models using the model framework registry
    logger.info("Fetching requested target models from centralized model registry: %s", model_names)
    models_dict = ModelRegistry.load(model_names)
    
    ## Initialize the optimized processing execution manager
    manager = ModelManager(models_dict=models_dict, batch_size=batch_size)

    # File batch processing loop
    for file_path in target_files:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(output_dir, file_name)
        
        logger.info("Starting batch validation operations for destination target: %s", file_name)

        try:
            ### Parse sequence payloads using the corrected dictionary hierarchy keys
            raw_data, unique_seqs = extract_unique_sequences(file_path)
            
            if not unique_seqs:
                logger.warn("Zero unique nucleotide structures isolated from file context: %s", file_name)
                continue

            logger.info("Isolated %s unique variants for high-throughput batch execution stream", len(unique_seqs))

            ### Execute high-performance multi-model cross scoring
            predictions_map = manager.predict_sequences(sequences=unique_seqs)

            ### Serialize updated metrics back to disk boundaries
            update_json_with_predictions(
                json_path=file_path,
                output_path=destination_path,
                predictions=predictions_map
            )
            
            logger.info("Successfully exported enriched structural dataset metrics log to: %s", destination_path)

        except Exception as error_exception:
            ### Capture full tracking traces for execution anomalies
            logger.error("Critical exceptional failure encountered on dataset target: %s", file_name, exc_info=True)
            raise error_exception