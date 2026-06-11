import pandas as pd
import numpy as np

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