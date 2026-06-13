# Heading 1 (Unified Analytical Entry-Point Loader)
## Structural JSON log flattening and metric integration pipeline
import json
import os
from pathlib import Path
from typing import List, Optional, Any
import pandas as pd
import numpy as np

from .utils.load_json import parse_json_folder
from .add_metrics.orchestror import calculate_trajectory_metrics
from ..utils.logger import get_custom_logger

# Instantiation Protocol
logger = get_custom_logger(__name__)



def OptimizerLoader(json_folder_path: Path | str, metrics_list: Optional[List[Any]] = None) -> pd.DataFrame:
    """
    High-performance modern loader to transform updated structural optimization 
    JSON file logs into metric-enriched DataFrames.

    :param json_folder_path: Path to the target directory containing optimization logs.
    :type json_folder_path: Path | str
    :param metrics_list: Optional collection subset filtering metric calculators.
    :type metrics_list: Optional[List[Any]]
    :return: Long-format DataFrame compiling chronological step metrics and beam states.
    :rtype: pd.DataFrame
    """
    logger.info("Executing log ingestion pass on path: %s", json_folder_path)
    
    # Flatten raw nested tree configurations directly to dataframe matrices
    rtn_df = parse_json_folder(json_folder_path)

    if rtn_df.empty:
        return rtn_df

    # Append downstream structural metrics
    logger.debug("Executing trajectory mathematical calculator post-processing loops.")
    if metrics_list:
        metric_df = calculate_trajectory_metrics(rtn_df, metrics_list)
    else:
        metric_df = calculate_trajectory_metrics(rtn_df)

    # Correlate calculated indicators back into the primary dataframe timeline
    rtn_df = pd.merge(
        rtn_df, 
        metric_df,
        on=['sequence_name', 'interpreter', 'optimizer', 'iteration'], 
        how='left'
    )
    return rtn_df