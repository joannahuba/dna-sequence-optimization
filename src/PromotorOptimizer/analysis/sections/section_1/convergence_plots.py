import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
from PromotorOptimizer.utils.logger import get_custom_logger

# Logging infrastructure initialization
logger = get_custom_logger(__name__)


# Core plotting infrastructure
## Low-level cell renderer agnostic of canvas layout

def draw_trajectory_cell(
    ax1: plt.Axes,
    df_isolated: pd.DataFrame,
    reconstruction_target_expression: float = None,
    config: Dict[str, Any] = None
) -> None:
    """
    Render mathematical trajectory curves onto a designated axes instance.

    :param ax1: Primary left-hand side matplotlib axes object.
    :type ax1: plt.Axes
    :param df_isolated: Pre-filtered DataFrame slice containing tracking metrics for one run.
    :type df_isolated: pd.DataFrame
    :param reconstruction_target_expression: Optional ground-truth expression level baseline, defaults to None.
    :type reconstruction_target_expression: float, optional
    :param config: Extensible configuration dictionary for visual adjustments, defaults to None.
    :type config: Dict[str, Any], optional
    :return: None
    """
    ## Extract optional parameters from configuration dictionary
    config = config or {}
    add_ensemble = config.get("add_ensemble_score", True)
    lw = config.get("linewidth", 1.4)
    
    ## Spatial mean aggregation across populations
    score_columns = ['score_deepstarr', 'score_deepstarr_second', 'score_original_modified', 'score_cost_function']
    trajectory_stats = df_isolated.groupby('iteration')[score_columns].mean().reset_index().sort_values('iteration')

    ## Map core validator models to primary coordinate system
    ax1.plot(trajectory_stats['iteration'], trajectory_stats['score_deepstarr'], color='#1f77b4', linestyle='--', linewidth=lw, label='Deepstar')
    ax1.plot(trajectory_stats['iteration'], trajectory_stats['score_deepstarr_second'], color='#ff7f0e', linestyle=':', linewidth=lw, label='2nd Deepstar')
    ax1.plot(trajectory_stats['iteration'], trajectory_stats['score_original_modified'], color='#2ca02c', linestyle='-.', linewidth=lw, label='Original')

    ## Render static target baseline bounds
    if reconstruction_target_expression is not None:
        ax1.axhline(y=reconstruction_target_expression, color='#7f7f7f', linestyle='-', linewidth=1.2, alpha=0.7)

    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    ## Orchestrate secondary y-axis mapping for joint metrics
    if add_ensemble:
        ax2 = ax1.twinx()
        ax2.plot(trajectory_stats['iteration'], trajectory_stats['score_cost_function'], color='#d62728', linestyle='-', linewidth=2.0, label='Ensemble')
        ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=8)

