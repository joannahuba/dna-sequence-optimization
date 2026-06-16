# Visualization pipeline component 1 (Updated for high visibility)
## Structural rendering grid for distribution metric properties with enhanced contrast

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# Plotting infrastructure definition
## Centralized multi-panel layout matrix generator

import matplotlib.pyplot as plt



def plot_unified_metric_grid(
    df_stats: pd.DataFrame,
    line1_suffix: str,
    line2_suffix: str,
    line1_latex_label: str,
    line2_latex_label: str,
    y_label_text: str,
    enforce_y_limit: bool = False
) -> plt.Figure:
    """
    Generate a unified high-contrast multi-panel matrix grid for any paired
    trajectory metrics using dynamic column suffix routing.

    :param df_stats: Aggregated metrics dataframe containing cross-sectional summaries.
    :type df_stats: pd.DataFrame
    :param line1_suffix: Column suffix for the first metric line type (e.g., 'var', 'stab_v').
    :type line1_suffix: str
    :param line2_suffix: Column suffix for the second metric line type (e.g., 'iqr', 'stab_i').
    :type line2_suffix: str
    :param line1_latex_label: LaTeX mathematical notation suffix string for line 1 legend.
    :type line1_latex_label: str
    :param line2_latex_label: LaTeX mathematical notation suffix string for line 2 legend.
    :type line2_latex_label: str
    :param y_label_text: Text label to apply to the outer vertical axes.
    :type y_label_text: str
    :param enforce_y_limit: Toggle to cap the vertical axis layout boundaries at [0, 1.05], defaults to False.
    :type enforce_y_limit: bool
    :return: Completely configured composite matplotlib Figure instance.
    :rtype: plt.Figure
    """
    # Layout orchestration block
    ## Extract unique environmental coordinates from the dataframe fields
    unique_interpreters = sorted(df_stats['interpreter'].unique())
    unique_optimizers = sorted(df_stats['optimizer'].unique())
    metrics = ['score_deepstarr', 'score_deepstarr_second', 'score_original_modified']
    
    color_map = {
        'score_deepstarr': '#1a73e8',
        'score_deepstarr_second': '#e67e22',
        'score_original_modified': '#2ea44f'
    }
    lbl_map = {
        'score_deepstarr': 'deepstarr',
        'score_deepstarr_second': 'deepstarr_second',
        'score_original_modified': 'original_modified'
    }
    
    fig, axes = plt.subplots(
        nrows=len(unique_optimizers),
        ncols=len(unique_interpreters),
        figsize=(16, 11),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    legend_handles = {}

    ## Traversal matching row-wise optimizers and column-wise interpreters
    for r_idx, opt in enumerate(unique_optimizers):
        for c_idx, interp in enumerate(unique_interpreters):
            ax = axes[r_idx, c_idx]
            
            ### Isolate cohort data subsets matching operational coordinate tokens
            mask = (
                (df_stats['interpreter'] == interp) & 
                (df_stats['optimizer'] == opt)
            )
            sub_df = df_stats[mask].sort_values('iteration')
            
            if sub_df.empty:
                continue
                
            for m in metrics:
                color = color_map[m]
                
                ### Construct concrete target column lookup strings dynamically from wide schema
                col1_mean = f"{m}_{line1_suffix}_mean"
                col1_std = f"{m}_{line1_suffix}_std"
                col2_mean = f"{m}_{line2_suffix}_mean"
                col2_std = f"{m}_{line2_suffix}_std"
                
                ### Render Line 1 trajectory (Dashed visualization layer)
                h1, = ax.plot(sub_df['iteration'], sub_df[col1_mean], color=color, linestyle='--', alpha=0.65, linewidth=1.8)
                ax.fill_between(sub_df['iteration'], np.maximum(0, sub_df[col1_mean] - sub_df[col1_std]), sub_df[col1_mean] + sub_df[col1_std], color=color, alpha=0.04)
                
                ### Render Line 2 trajectory (Bold solid visualization layer)
                h2, = ax.plot(sub_df['iteration'], sub_df[col2_mean], color=color, linestyle='-', alpha=0.95, linewidth=3.0)
                ax.fill_between(sub_df['iteration'], np.maximum(0, sub_df[col2_mean] - sub_df[col2_std]), sub_df[col2_mean] + sub_df[col2_std], color=color, alpha=0.18)
                
                if lbl_map[m] not in legend_handles:
                    legend_handles[lbl_map[m]] = (h1, h2)
                    
            if enforce_y_limit:
                ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.45)
            
    ## Apply global border text annotations layout rules
    for col_idx, interp in enumerate(unique_interpreters):
        axes[0, col_idx].set_title(f"Interpreter: {interp}", fontsize=12, fontweight='bold', pad=12)
    for r_idx, opt in enumerate(unique_optimizers):
        axes[r_idx, 0].set_ylabel(f"Optimizer: {opt}\n\n{y_label_text}", fontsize=10, fontweight='bold', labelpad=15)
    for col_idx in range(len(unique_interpreters)):
        axes[-1, col_idx].set_xlabel("Optimization Iterations", fontsize=11, fontweight='bold', labelpad=10)

    ## Flatten legend tracking matrices into standard unified lists
    handles, labels = [], []
    for k, (h_v, h_i) in legend_handles.items():
        handles.extend([h_v, h_i])
        labels.extend([f"{k} ({line1_latex_label})", f"{k} ({line2_latex_label})"])
        
    fig.legend(handles=handles, labels=labels, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=10, frameon=True, facecolor='white', framealpha=0.95)
    plt.subplots_adjust(left=0.09, right=0.83, top=0.93, bottom=0.07, hspace=0.15, wspace=0.15)
    return fig