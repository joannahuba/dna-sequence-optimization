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
    # Configuration extraction block
    ## Extract architectural and layout configurations from supplement dictionary
    config = config or {}
    add_ensemble = config.get("add_ensemble_score_column", False)
    lw = config.get("linewidth", 1.4)
    
    ## Copy score columns slice to prevent downstream dictionary mutations
    score_columns = list(config.get(
        'score_columns', 
        ['score_deepstarr', 'score_deepstarr_second', 'score_original_modified', 'score_cost_function']
    ))
    
    # Dynamic column injection
    ## Append consolidated compound score target if explicitly defined in runtime configuration
    if isinstance(add_ensemble, str) and add_ensemble in df_isolated.columns and add_ensemble not in score_columns:
        logger.debug("Injecting external ensemble target column: %s into statistical loop", add_ensemble)
        score_columns.append(add_ensemble)
        
    # Spatial data processing
    ## Aggregate population snapshots chronologically using grouped mean statistics
    trajectory_stats = df_isolated.groupby('iteration')[score_columns].mean().reset_index().sort_values('iteration')

    # Mapping target columns dynamically
    ## Map target models based on positional indices to bypass hardcoded name requirements
    col_deepstarr = score_columns[0]
    col_deepstarr_second = score_columns[1]
    col_original_modified = score_columns[2]

    # Coordinate system rendering execution
    ## Plot structural model variations on primary axes interface
    ax1.plot(trajectory_stats['iteration'], trajectory_stats[col_deepstarr], color='#1f77b4', linestyle='--', linewidth=lw, label='Deepstar')
    ax1.plot(trajectory_stats['iteration'], trajectory_stats[col_deepstarr_second], color='#ff7f0e', linestyle=':', linewidth=lw, label='2nd Deepstar')
    ax1.plot(trajectory_stats['iteration'], trajectory_stats[col_original_modified], color='#2ca02c', linestyle='-.', linewidth=lw, label='Original')

    # Boundary overlays
    ## Render static target baseline bounds if present
    if reconstruction_target_expression is not None:
        ax1.axhline(y=reconstruction_target_expression, color='#7f7f7f', linestyle='-', linewidth=1.2, alpha=0.7)

    ## Plot ensemble metric directly on the primary axis to preserve shared scale execution
    if isinstance(add_ensemble, str) and add_ensemble in trajectory_stats.columns:
        ax1.plot(trajectory_stats['iteration'], trajectory_stats[add_ensemble], color='#d62728', linestyle='-', linewidth=2.0, label='Ensemble score')

    ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)