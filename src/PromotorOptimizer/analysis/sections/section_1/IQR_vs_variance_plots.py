
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


############################################################################################
# Plots for simulating paths comparing IQR vs \sigma^2 stability across long period (variance of variants)
############################################################################################



# def plot_robust_distribution_comparison(df, window=3, num_simulated_paths=15):
#     """
#     Simulates continuous optimization trajectories from unlinked beam data,
#     computes rolling IQR along each path, and plots cross-trajectory distributions.

#     :param df: Input DataFrame with raw mixed beam entries.
#     :type df: pandas.DataFrame
#     :param window: Window size for rolling IQR calculation.
#     :type window: int
#     :param num_simulated_paths: Number of stochastic paths to reconstruct.
#     :type num_simulated_paths: int
#     """
#     # Trajectory simulation block
#     ## Extract groups to isolate parallel runs
#     df = df.sort_values(['interpreter', 'optimizer', 'model_type', 'sequence_name', 'iteration']).copy()
#     config_groups = df.groupby(['interpreter', 'optimizer', 'model_type', 'sequence_name'])
#     all_simulated_tracks = []
    
#     ## Reconstruct sequential paths by stochastic sampling across iterations
#     for (interp, opt, model, seq), group in config_groups:
#         unique_iters = sorted(group['iteration'].unique())
        
#         for path_idx in range(num_simulated_paths):
#             path_rows = []
#             for it in unique_iters:
#                 iter_candidates = group[group['iteration'] == it]
#                 if iter_candidates.empty:
#                     continue
#                 ### Randomly sample one candidate state per iteration to build a continuous track
#                 sampled_row = iter_candidates.sample(n=1).iloc[0].copy()
#                 sampled_row['path_id'] = f"{seq}_path_{path_idx}"
#                 path_rows.append(sampled_row)
                
#             if path_rows:
#                 path_df = pd.DataFrame(path_rows).sort_values('iteration')
#                 ### Calculate rolling IQR over this specific isolated track
#                 raw_var = path_df['variance_individual']
#                 path_df['rolling_iqr_variance'] = (
#                     raw_var.rolling(window=window, min_periods=1).quantile(0.75) - 
#                     raw_var.rolling(window=window, min_periods=1).quantile(0.25)
#                 )
#                 all_simulated_tracks.append(path_df)

#     # Statistical cross-sectional aggregation
#     ## Combine all paths and aggregate mean/std at each iteration timestamp
#     master_df = pd.concat(all_simulated_tracks, ignore_index=True)
#     df_stats = master_df.groupby(['interpreter', 'optimizer', 'model_type', 'iteration']).agg(
#         raw_mean=('variance_individual', 'mean'),
#         raw_std=('variance_individual', 'std'),
#         iqr_mean=('rolling_iqr_variance', 'mean'),
#         iqr_std=('rolling_iqr_variance', 'std')
#     ).reset_index()

#     # Visualization execution matching requested aesthetics
#     unique_interpreters = df_stats['interpreter'].unique()
#     unique_optimizers = df_stats['optimizer'].unique()
#     unique_validators = df_stats['model_type'].unique()
    
#     base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#     color_mapping = {val: base_colors[i] for i, val in enumerate(unique_validators)}
    
#     fig, axes = plt.subplots(len(unique_optimizers), len(unique_interpreters), 
#                              figsize=(15, 10), sharex=True, squeeze=False)
    
#     for row_idx, opt in enumerate(unique_optimizers):
#         for col_idx, interp in enumerate(unique_interpreters):
#             ax = axes[row_idx, col_idx]
#             sub_df = df_stats[(df_stats['interpreter'] == interp) & (df_stats['optimizer'] == opt)]
            
#             for val in unique_validators:
#                 v_df = sub_df[sub_df['model_type'] == val].sort_values('iteration')
#                 if v_df.empty:
#                     continue
                
#                 base_color = color_mapping[val]
                
#                 # Plot Raw Variance Space Distribution
#                 ax.plot(v_df['iteration'], v_df['raw_mean'], 
#                         color=base_color, linestyle=':', alpha=0.5, linewidth=1.2,
#                         label=f"{val} (Raw Var Mean)")
#                 ax.fill_between(v_df['iteration'], 
#                                 np.maximum(0, v_df['raw_mean'] - v_df['raw_std']), 
#                                 v_df['raw_mean'] + v_df['raw_std'], 
#                                 color=base_color, alpha=0.08, label='_nolegend_')
                
#                 # Plot Robust Rolling IQR Space Distribution
#                 ax.plot(v_df['iteration'], v_df['iqr_mean'], 
#                         color=base_color, linestyle='-', alpha=0.9, linewidth=2.5,
#                         label=f"{val} (Rolling IQR Mean)")
#                 ax.fill_between(v_df['iteration'], 
#                                 np.maximum(0, v_df['iqr_mean'] - v_df['iqr_std']), 
#                                 v_df['iqr_mean'] + v_df['iqr_std'], 
#                                 color=base_color, alpha=0.25, label='_nolegend_')
            
#             ax.set_title(f"Interpreter: {interp} | Optimizer: {opt}", fontsize=11, fontweight='bold')
#             if row_idx == len(unique_optimizers) - 1:
#                 ax.set_xlabel("Optimization Iterations", fontweight='bold')
#             if col_idx == 0:
#                 ax.set_ylabel("Metric Distribution Spread", fontweight='bold')
#             if row_idx == 0 and col_idx == 0:
#                 ax.legend(loc='upper right', frameon=True, fontsize=8, facecolor='white', framealpha=0.9)
                
#     plt.tight_layout()
#     # plt.savefig('final_adapted_distribution_grid.png', dpi=300)
#     plt.show()
#     plt.close()




# TODO sprawdzić - ten powinien być dziłąjcy
def plot_robust_distribution_comparison(df, window=3, num_simulated_paths=15):
    """
    Simulates continuous optimization trajectories from unlinked beam data,
    computes rolling IQR along each path, and plots the cross-trajectory distributions.
    
    Overlays Raw Variance (lighter hue, thinner dotted line) and Rolling IQR (darker hue, 
    bold solid line) simultaneously for each validator model on a structured 4x3 grid.
    Layout labels are pushed strictly to the outer boundaries of the global grid.

    :param df: Input DataFrame with raw mixed beam entries.
    :type df: pandas.DataFrame
    :param window: Window size for rolling IQR calculation.
    :type window: int
    :param num_simulated_paths: Number of stochastic paths to reconstruct.
    :type num_simulated_paths: int
    """
    # Trajectory simulation block
    ## Extract groups to isolate parallel runs
    df = df.sort_values(['interpreter', 'optimizer', 'val_model', 'sequence_name', 'iteration']).copy()
    config_groups = df.groupby(['interpreter', 'optimizer', 'val_model', 'sequence_name'])
    all_simulated_tracks = []
    
    ## Reconstruct sequential paths by stochastic sampling across iterations
    for (interp, opt, model, seq), group in config_groups:
        unique_iters = sorted(group['iteration'].unique())
        
        for path_idx in range(num_simulated_paths):
            path_rows = []
            for it in unique_iters:
                iter_candidates = group[group['iteration'] == it]
                if iter_candidates.empty:
                    continue
                ### Randomly sample one candidate state per iteration to build a continuous track
                sampled_row = iter_candidates.sample(n=1).iloc[0].copy()
                sampled_row['path_id'] = f"{seq}_path_{path_idx}"
                path_rows.append(sampled_row)
                
            if path_rows:
                path_df = pd.DataFrame(path_rows).sort_values('iteration')
                ### Calculate rolling IQR over this specific isolated track
                raw_var = path_df['variance_individual']
                path_df['rolling_iqr_variance'] = (
                    raw_var.rolling(window=window, min_periods=1).quantile(0.75) - 
                    raw_var.rolling(window=window, min_periods=1).quantile(0.25)
                )
                all_simulated_tracks.append(path_df)

    # Statistical cross-sectional aggregation
    ## Combine all paths and aggregate mean/std at each iteration timestamp
    master_df = pd.concat(all_simulated_tracks, ignore_index=True)
    df_stats = master_df.groupby(['interpreter', 'optimizer', 'val_model', 'iteration']).agg(
        raw_mean=('variance_individual', 'mean'),
        raw_std=('variance_individual', 'std'),
        iqr_mean=('rolling_iqr_variance', 'mean'),
        iqr_std=('rolling_iqr_variance', 'std')
    ).reset_index()

    # Structural grid configuration (4 Optimizers x 3 Interpreters)
    unique_interpreters = sorted(df_stats['interpreter'].unique())
    unique_optimizers = sorted(df_stats['optimizer'].unique())
    unique_validators = sorted(df_stats['val_model'].unique())
    
    ## Establish a single global scale across all distributions to force uniform y-limits
    global_max_scale = max(
        (df_stats['raw_mean'] + df_stats['raw_std']).max(),
        (df_stats['iqr_mean'] + df_stats['iqr_std']).max()
    )
    
    ## Define color standards for 3 distinct models
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color_mapping = {val: base_colors[i % len(base_colors)] for i, val in enumerate(unique_validators)}
    
    ## Figure layout architecture setup
    fig, axes = plt.subplots(len(unique_optimizers), len(unique_interpreters), 
                             figsize=(16, 11), sharex=True, sharey=True, squeeze=False)
    
    ## Retain plot handles exclusively for the global unified legend design
    legend_handles = {}

    # Main grid layout rendering loop
    for row_idx, opt in enumerate(unique_optimizers):
        for col_idx, interp in enumerate(unique_interpreters):
            ax = axes[row_idx, col_idx]
            sub_df = df_stats[(df_stats['interpreter'] == interp) & (df_stats['optimizer'] == opt)]
            
            ## Draw overlay metrics for each specific validator model
            for val in unique_validators:
                v_df = sub_df[sub_df['val_model'] == val].sort_values('iteration')
                if v_df.empty:
                    continue
                
                c = color_mapping[val]
                
                ### Case 1: Raw Variance (Lighter hue, thinner dotted line, low opacity area)
                h_raw, = ax.plot(v_df['iteration'], v_df['raw_mean'], 
                                 color=c, linestyle=':', alpha=0.40, linewidth=1.2)
                ax.fill_between(v_df['iteration'], 
                                np.maximum(0, v_df['raw_mean'] - v_df['raw_std']), 
                                v_df['raw_mean'] + v_df['raw_std'], 
                                color=c, alpha=0.03)
                
                ### Case 2: Rolling IQR (Darker hue, bold solid line, high opacity area)
                h_iqr, = ax.plot(v_df['iteration'], v_df['iqr_mean'], 
                                 color=c, linestyle='-', alpha=0.95, linewidth=2.4)
                ax.fill_between(v_df['iteration'], 
                                np.maximum(0, v_df['iqr_mean'] - v_df['iqr_std']), 
                                v_df['iqr_mean'] + v_df['iqr_std'], 
                                color=c, alpha=0.18)
                
                ### Store unique plot instances for global labeling execution
                if val not in legend_handles:
                    legend_handles[val] = (h_raw, h_iqr)
            
            ## Enforce uniform axis constraints across the complete canvas matrix
            ax.set_ylim(0, global_max_scale * 1.05)
            ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # Outer grid boundary decoration block
    ## Layout optimization: place titles strictly on outer borders to maintain cleanliness
    for col_idx, interp in enumerate(unique_interpreters):
        axes[0, col_idx].set_title(f"Interpreter: {interp}", fontsize=12, fontweight='bold', pad=12)
        
    for row_idx, opt in enumerate(unique_optimizers):
        ### Rotate and center the outer row definitions on the left-hand margin
        axes[row_idx, 0].set_ylabel(f"Optimizer: {opt}\n\nMetric Spread", 
                                    fontweight='bold', fontsize=10, rotation=90, labelpad=15)
        
    for col_idx in range(len(unique_interpreters)):
        axes[-1, col_idx].set_xlabel("Optimization Iterations", fontweight='bold', fontsize=10, labelpad=10)

    # Global legend construction
    ## Flattens handles and explicit labels into standard lists
    final_handles = []
    final_labels = []
    for val in unique_validators:
        if val in legend_handles:
            h_raw, h_iqr = legend_handles[val]
            
            final_handles.extend([h_raw, h_iqr])
            final_labels.extend([f"{val} ($\sigma^2$)", f"{val} (IQR)"])

    ## Render a single master legend block outside the axis boundary maps
    fig.legend(handles=final_handles, labels=final_labels, loc='center right', 
               bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=True, facecolor='white', framealpha=0.95)

    ## Refactor geometric space allocation parameters to fit the global legend layout
    plt.subplots_adjust(left=0.08, right=0.86, top=0.93, bottom=0.07, hspace=0.12, wspace=0.12)
    plt.show()
    plt.close()





