

from .utils.json_to_df import aggregate_advanced_trajectory
from .utils.load_json import parse_folder
from .add_metrics.orchestror import calculate_trajectory_metrics

from pathlib import Path
import pandas as pd

def OptimizerLoader(json_folder_path: Path | str, metrics_list: list | None = None):

    # load
    ## json -> dict
    strange_dict = parse_folder(json_folder_path)
    ## dict -> pd.DataFrame
    rtn_df = aggregate_advanced_trajectory(strange_dict)

    # additional calculations 
    ## case: load specific metrics
    if metrics_list:
        metric_df = calculate_trajectory_metrics(rtn_df, metrics_list)
    ## case: default=load all metrics
    else:
        metric_df = calculate_trajectory_metrics(rtn_df)

    rtn_df = pd.merge(
        rtn_df, 
        metric_df,
        on=['sequence_name', 'interpreter', 'optimizer', 'iteration'], 
        how='left'
    )
    

    return rtn_df