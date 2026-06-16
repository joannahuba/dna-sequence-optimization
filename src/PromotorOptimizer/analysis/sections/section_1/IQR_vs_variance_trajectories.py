import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

from ....utils.logger import get_custom_logger
# Logging infrastructure initialization
logger = get_custom_logger(__name__)

# Data processing and path simulation engine
## Reconstruct continuous trajectory tracks from beam populations
import numpy as np
import pandas as pd
from typing import List

from ....utils.logger import get_custom_logger

# Logging infrastructure initialization
logger = get_custom_logger(__name__)

# Data processing and path simulation engine
## Reconstruct continuous trajectory tracks from beam populations

def simulate_trajectory_paths(
    df: pd.DataFrame, 
    num_simulated_paths: int = 15
) -> pd.DataFrame:
    """
    Reconstruct continuous chronological optimization paths by sampling adjacent 
    iteration states from the configuration cohorts after injecting population beam means.

    :param df: Input DataFrame with raw mixed beam entries.
    :type df: pd.DataFrame
    :param num_simulated_paths: Number of parallel stochastic tracks to sample.
    :type num_simulated_paths: int
    :return: DataFrame containing individual simulated paths with inline beam mean metrics.
    :rtype: pd.DataFrame
    """
    # Data initialization block
    ## Create a working copy and define structural grouping coordinates
    df_working = df.copy()
    group_keys = ['interpreter', 'optimizer', 'model_type', 'sequence_name', 'iteration']
    score_cols = ['score_deepstarr', 'score_deepstarr_second', 'score_original_modified']

    # Population statistics processing
    ## Pre-compute beam mean for each target validation metric across the population slice
    for col in score_cols:
        logger.debug("Pre-computing population beam mean for column: %s", col)
        df_working[f'beam_mean_{col}'] = df_working.groupby(group_keys)[col].transform('mean')

    # Spatial sorting preparation
    ## Sort database records chronologically before executing cohort splits
    df_working = df_working.sort_values(['interpreter', 'optimizer', 'model_type', 'sequence_name', 'iteration'])
    cohort_groups = df_working.groupby(['interpreter', 'optimizer', 'model_type', 'sequence_name'])
    all_sampled_paths = []

    # Path reconstruction loop
    ## Isolate unlinked beams and stochastically assemble continuous linear tracks
    for (interp, opt, model, seq), group in cohort_groups:
        unique_iters = sorted(group['iteration'].unique())
        
        for path_idx in range(num_simulated_paths):
            path_rows = []
            for it in unique_iters:
                iter_candidates = group[group['iteration'] == it]
                if iter_candidates.empty:
                    continue
                ### Sample a single candidate row per iteration to preserve path context
                sampled_row = iter_candidates.sample(n=1).iloc[0].copy()
                sampled_row['path_id'] = f"{interp}_{opt}_{model}_{seq}_path_{path_idx}"
                path_rows.append(sampled_row)
                
            if path_rows:
                all_sampled_paths.append(pd.DataFrame(path_rows).sort_values('iteration'))

    if not all_sampled_paths:
        raise ValueError("Failed to isolate path tracks. Verify data layout keys.")
        
    return pd.concat(all_sampled_paths, ignore_index=True)

# Statistical transformation engine
## Apply localized rolling window properties and stability index map transformations

def compute_stability_indices(
    df_paths: pd.DataFrame, 
    window: int = 3, 
    gamma: float = 1.0
) -> pd.DataFrame:
    """
    Compute variance cumulatively over time (expanding window) and IQR over a 
    rolling window along each individual trajectory path.
    """
    score_cols = ['score_deepstarr', 'score_deepstarr_second', 'score_original_modified']
    path_groups = df_paths.groupby('path_id')
    processed_tracks = []

    for path_id, path_df in path_groups:
        path_df = path_df.sort_values('iteration').copy()
        
        for col in score_cols:
            ### Fix: Compute variance cumulatively over time up to step t along the path
            expanding_var = path_df[col].expanding(min_periods=1).var().fillna(0)
            path_df[f'var_{col}'] = expanding_var
            
            ### Compute rolling IQR of that variance series over window W
            q3 = expanding_var.rolling(window=window, min_periods=1).quantile(0.75)
            q1 = expanding_var.rolling(window=window, min_periods=1).quantile(0.25)
            iqr_series = q3 - q1
            path_df[f'iqr_{col}'] = iqr_series
            
            ### Mathematical transformations into stability space
            path_df[f'stab_var_{col}'] = 1.0 / (1.0 + expanding_var)
            path_df[f'stab_iqr_{col}'] = np.exp(-gamma * (iqr_series ** 2))

            epsilon = 1e-9
            
            # Extract beam mean for normalization from the source dataframe population slice
            ## (Calculated directly from the current step's population prior to sampling)
            beam_mu = np.abs(path_df[f'beam_mean_{col}']) + epsilon

            # 1. Relative variance calculation
            ## Column name format: f'var_rel_{col}'
            expanding_var_rel = expanding_var / (beam_mu ** 2)
            path_df[f'var_rel_{col}'] = expanding_var_rel

            # 2. Relative rolling IQR calculation
            ## Column name format: f'iqr_rel_{col}'
            q3_rel = expanding_var_rel.rolling(window=window, min_periods=1).quantile(0.75)
            q1_rel = expanding_var_rel.rolling(window=window, min_periods=1).quantile(0.25)
            iqr_series_rel = q3_rel - q1_rel
            path_df[f'iqr_rel_{col}'] = iqr_series_rel

            # 3. Relative variance stability index transformation
            ## Column name format: f'stab_var_rel_{col}'
            path_df[f'stab_var_rel_{col}'] = 1.0 / (1.0 + np.sqrt(expanding_var_rel))

            # 4. Relative IQR stability index transformation
            ## Column name format: f'stab_iqr_rel_{col}'
            path_df[f'stab_iqr_rel_{col}'] = np.exp(-gamma * (iqr_series_rel ** 2))
            
        processed_tracks.append(path_df)

    master_tracks = pd.concat(processed_tracks, ignore_index=True)
    
    # Cross-sectional aggregation across simulated trajectories
    agg_rules = {}
    # Cross-sectional aggregation across simulated trajectories
    ## Expanded aggregation dictionary to preserve both raw and relative statistics
    agg_rules = {}
    for col in score_cols:
        ### Base raw metrics definitions
        agg_rules[f'{col}_var_mean'] = (f'var_{col}', 'mean')
        agg_rules[f'{col}_var_std'] = (f'var_{col}', 'std')
        agg_rules[f'{col}_iqr_mean'] = (f'iqr_{col}', 'mean')
        agg_rules[f'{col}_iqr_std'] = (f'iqr_{col}', 'std')
        agg_rules[f'{col}_stab_v_mean'] = (f'stab_var_{col}', 'mean')
        agg_rules[f'{col}_stab_v_std'] = (f'stab_var_{col}', 'std')
        agg_rules[f'{col}_stab_i_mean'] = (f'stab_iqr_{col}', 'mean')
        agg_rules[f'{col}_stab_i_std'] = (f'stab_iqr_{col}', 'std')
        
        ### Normalized relative metrics definitions
        agg_rules[f'{col}_var_rel_mean'] = (f'var_rel_{col}', 'mean')
        agg_rules[f'{col}_var_rel_std'] = (f'var_rel_{col}', 'std')
        agg_rules[f'{col}_iqr_rel_mean'] = (f'iqr_rel_{col}', 'mean')
        agg_rules[f'{col}_iqr_rel_std'] = (f'iqr_rel_{col}', 'std')
        agg_rules[f'{col}_stab_var_rel_mean'] = (f'stab_var_rel_{col}', 'mean')
        agg_rules[f'{col}_stab_var_rel_std'] = (f'stab_var_rel_{col}', 'std')
        agg_rules[f'{col}_stab_iqr_rel_mean'] = (f'stab_iqr_rel_{col}', 'mean')
        agg_rules[f'{col}_stab_iqr_rel_std'] = (f'stab_iqr_rel_{col}', 'std')

    return master_tracks.groupby(['interpreter', 'optimizer', 'iteration']).agg(**agg_rules).reset_index()

    return master_tracks.groupby(['interpreter', 'optimizer', 'iteration']).agg(**agg_rules).reset_index()


#TODO - to jest dobre ale upewnić się że to liczy wariancje po wszystkich wariancjach do danego kroku t
def generate_robust_analysis_table(df, window=3, num_simulated_paths=10):
    """
    Simulates continuous optimization trajectories across beam states, computes localized 
    scale metrics along each path, and cross-tabulates the resulting statistical 
    distributions across interpreters and optimizers.

    :param df: Cleaned input DataFrame containing optimization iterations and individual variances.
    :type df: pandas.DataFrame
    :param window: Temporal window size $w$ for the rolling IQR calculation.
    :type window: int
    :param num_simulated_paths: Number of stochastic paths to sample when parent-child lineage is missing.
    :type num_simulated_paths: int
    :return: A pivot table showing the mean and peak reduction across interpreters (columns) and optimizers (rows).
    :rtype: pandas.DataFrame
    """
    # Initialize path storage list
    all_simulated_tracks = []
    
    # Extract unique environmental configurations
    config_groups = df.groupby(['interpreter', 'optimizer', 'model_type', 'sequence_name'])
    
    for (interp, opt, model, seq), group in config_groups:
        ## Extract sorted unique iterations for the current sequence
        unique_iters = sorted(group['iteration'].unique())
        if len(unique_iters) < window:
            continue
            
        ## Reconstruct independent trajectories across the beam width
        for path_idx in range(num_simulated_paths):
            path_rows = []
            
            for it in unique_iters:
                ### Sample a random candidate state per iteration to simulate a continuous path
                iter_candidates = group[group['iteration'] == it]
                if iter_candidates.empty:
                    continue
                sampled_row = iter_candidates.sample(n=1).iloc[0].copy()
                sampled_row['path_id'] = f"{seq}_path_{path_idx}"
                path_rows.append(sampled_row)
                
            ### Compile path rows into a temporary dataframe to evaluate metrics along the time axis
            if path_rows:
                path_df = pd.DataFrame(path_rows).sort_values('iteration')
                
                # Calculate metrics along the simulated trajectory
                ## Compute raw variance properties along the path
                raw_var = path_df['variance_individual']
                
                ## Compute the robust rolling IQR metric over the localized window w
                rolling_iqr = (
                    raw_var.rolling(window=window, min_periods=1).quantile(0.75) - 
                    raw_var.rolling(window=window, min_periods=1).quantile(0.25)
                )
                
                path_df['rolling_iqr_variance'] = rolling_iqr
                all_simulated_tracks.append(path_df)

    # Aggregate distribution properties across all simulated trajectories
    ## Concatenate all tracks into a structural master frame
    if not all_simulated_tracks:
        return pd.DataFrame()
        
    master_paths_df = pd.concat(all_simulated_tracks, ignore_index=True)
    
    ## Calculate distribution summaries (Mean and Maximum Peak values) per track
    summary_df = master_paths_df.groupby(['interpreter', 'optimizer', 'model_type', 'path_id']).agg(
        mean_raw_variance=('variance_individual', 'mean'),
        max_raw_variance=('variance_individual', 'max'),
        mean_rolling_iqr=('rolling_iqr_variance', 'mean'),
        max_rolling_iqr=('rolling_iqr_variance', 'max')
    ).reset_index()
    
    ## Quantify the distribution shift (mitigation performance)
    summary_df['peak_mitigation_pct'] = (
        (summary_df['max_raw_variance'] - summary_df['max_rolling_iqr']) / summary_df['max_raw_variance'] * 100
    )
    
    # Construct the final matrix presentation
    ## Group data to separate validation models and compute overall macro statistics
    final_agg = summary_df.groupby(['optimizer', 'interpreter', 'model_type']).agg(
        avg_raw_peak=('max_raw_variance', 'mean'),
        avg_iqr_peak=('max_rolling_iqr', 'mean'),
        mitigation_eff=('peak_mitigation_pct', 'mean')
    ).reset_index()
    
    ## Format cells to output a comprehensive string representation: Peak Var -> Peak IQR (Mitigation %)
    final_agg['cell_value'] = final_agg.apply(
        lambda r: f"{r['avg_raw_peak']:.2f} → {r['avg_iqr_peak']:.2f} (-{r['mitigation_eff']:.1f}%)", axis=1
    )
    
    ## Pivot the table to position Interpreters as columns and Optimizers/Model Types as rows
    pivot_matrix = final_agg.pivot(
        index=['optimizer', 'model_type'], 
        columns='interpreter', 
        values='cell_value'
    ).fillna("N/A")
    
    return pivot_matrix