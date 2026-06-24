import os
import matplotlib.pyplot as plt
import pandas as pd
from PromotorOptimizer.utils.logger import get_custom_logger

# Logging infrastructure initialization
logger = get_custom_logger(__name__)


# Advanced visualization modules with centralized global legend matrices
## High-level plotting routines with multi-axis coordinate alignment

def plot_shannon_entropy_panels(
    df_metrics: pd.DataFrame, 
    score_col: str = 'final_adjusted_score', 
    save_path: str = None
) -> None:
    """
    Generate a multi-panel figure displaying Shannon Entropy trajectories alongside 
    the consolidated ensemble score with a unified global legend.

    :param df_metrics: DataFrame containing calculated metrics and tracking tracks.
    :type df_metrics: pd.DataFrame
    :param score_col: Name of the consolidated target score column to project.
    :type score_col: str
    :param save_path: Optional file path to save the generated composite figure.
    :type save_path: str, optional
    :return: None
    """
    # Summary aggregation block
    ## Compute mean and standard error across sequence replicates for both metrics simultaneously
    logger.debug("Aggregating summary statistics for target score column: %s", score_col)
    group_keys = ['optimizer', 'interpreter', 'iteration']
    agg_dict = {
        'shannon_entropy': ['mean', 'sem'],
        score_col: ['mean', 'sem']
    }
    
    df_summary = df_metrics.groupby(group_keys).agg(agg_dict)
    df_summary.columns = [f"{col}_{stat}" for col, stat in df_summary.columns]
    df_summary = df_summary.reset_index()
    
    # Global scale harmonization
    ## Pre-calculate global score boundaries to ensure uniform limits across independent twin axes
    score_min = df_summary[f"{score_col}_mean"].min()
    score_max = df_summary[f"{score_col}_mean"].max()
    score_pad = (score_max - score_min) * 0.08 if score_max != score_min else 1.0

    # Grid initialization
    ## Identify unique optimizers to define the subplot layout dynamically
    unique_optimizers = df_summary['optimizer'].unique()
    num_plots = len(unique_optimizers)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5.5 * num_plots, 4.8), sharey=True)
    if num_plots == 1:
        axes = [axes]
        
    global_handles = {}
    
    # Multi-panel execution
    ## Iterate over each optimizer to build localized axes
    for idx, opt_name in enumerate(unique_optimizers):
        ax = axes[idx]
        opt_data = df_summary[df_summary['optimizer'] == opt_name]
        
        ## Generate shared coordinate twin system for score overlays
        ax_twin = ax.twinx()
        ax_twin.set_ylim(score_min - score_pad, score_max + score_pad)
        ax_twin.spines['right'].set_color('#555555')
        ax_twin.tick_params(axis='y', colors='#555555', labelsize=8)
        
        ## Plot distinct interpreter trajectories inside the current optimizer context
        unique_interpreters = opt_data['interpreter'].unique()
        colors = plt.cm.tab10(range(len(unique_interpreters)))
        
        for interp_idx, interp_name in enumerate(unique_interpreters):
            interp_data = opt_data[opt_data['interpreter'] == interp_name]
            color = colors[interp_idx]
            
            ### Render mean line trajectory for primary Shannon Entropy
            h_line, = ax.plot(
                interp_data['iteration'], 
                interp_data['shannon_entropy_mean'], 
                color=color, 
                lw=2, 
                linestyle='-'
            )
            ax.fill_between(
                interp_data['iteration'], 
                interp_data['shannon_entropy_mean'] - interp_data['shannon_entropy_sem'], 
                interp_data['shannon_entropy_mean'] + interp_data['shannon_entropy_sem'], 
                color=color, 
                alpha=0.12
            )
            
            ### Render corresponding consolidated score profile on the secondary vertical axis
            s_line, = ax_twin.plot(
                interp_data['iteration'], 
                interp_data[f"{score_col}_mean"], 
                color=color, 
                lw=1.5, 
                linestyle='--',
                alpha=0.85
            )
            ax_twin.fill_between(
                interp_data['iteration'], 
                interp_data[f"{score_col}_mean"] - interp_data[f"{score_col}_sem"], 
                interp_data[f"{score_col}_mean"] + interp_data[f"{score_col}_sem"], 
                color=color, 
                alpha=0.03
            )
            
            ### Cache handles for the unified global legend assembly
            if f"{interp_name} ($H$)" not in global_handles:
                global_handles[f"{interp_name} ($H$)"] = h_line
            if f"{interp_name} (Score)" not in global_handles:
                global_handles[f"{interp_name} (Score)"] = s_line
            
        ## Format axes styling and mathematical labels
        ax.set_title(f"Optimizer: {opt_name}", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Iteration ($t$)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if idx == 0:
            ax.set_ylabel("Shannon Entropy ($H(t)$)", fontsize=10, fontweight='bold')
            
        if idx == num_plots - 1:
            ax_twin.set_ylabel("Consolidated Ensemble Score", fontsize=10, fontweight='bold', color='#333333', labelpad=10)
        else:
            ax_twin.set_yticklabels([])

    # Centralized layout processing
    ## Render single global horizontal legend beneath the subplots canvas
    fig.legend(
        list(global_handles.values()), 
        list(global_handles.keys()), 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.06), 
        ncol=len(global_handles), 
        frameon=True,
        fontsize=8.5
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Save framework output
    if save_path:
        logger.info("Saving consolidated entropy panel report to: %s", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hamming_trajectory_panels(
    df_metrics: pd.DataFrame, 
    score_col: str = 'final_adjusted_score', 
    save_path: str = None
) -> None:
    """
    Generate a multi-panel figure displaying Cumulative Normalized Hamming Distance alongside 
    the consolidated ensemble score with a unified global legend.

    :param df_metrics: DataFrame containing calculated metrics from Module 1.
    :type df_metrics: pd.DataFrame
    :param score_col: Name of the consolidated target score column to project.
    :type score_col: str
    :param save_path: Optional file path to save the generated composite figure.
    :type save_path: str, optional
    :return: None
    """
    # Summary aggregation block
    ## Compute mean and standard error across sequence replicates for both metrics simultaneously
    logger.debug("Aggregating summary statistics for target score column: %s", score_col)
    group_keys = ['interpreter', 'optimizer', 'iteration']
    agg_dict = {
        'hamming_trajectory': ['mean', 'sem'],
        score_col: ['mean', 'sem']
    }
    
    df_summary = df_metrics.groupby(group_keys).agg(agg_dict)
    df_summary.columns = [f"{col}_{stat}" for col, stat in df_summary.columns]
    df_summary = df_summary.reset_index()
    
    # Global scale harmonization
    ## Pre-calculate global score boundaries to ensure uniform limits across independent twin axes
    score_min = df_summary[f"{score_col}_mean"].min()
    score_max = df_summary[f"{score_col}_mean"].max()
    score_pad = (score_max - score_min) * 0.08 if score_max != score_min else 1.0

    # Grid initialization
    ## Identify unique interpreters to define the subplot layout dynamically
    unique_interpreters = df_summary['interpreter'].unique()
    num_plots = len(unique_interpreters)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5.5 * num_plots, 4.8), sharey=True)
    if num_plots == 1:
        axes = [axes]
        
    global_handles = {}
    
    # Multi-panel execution
    ## Iterate over each interpreter to build localized axes
    for idx, interp_name in enumerate(unique_interpreters):
        ax = axes[idx]
        interp_data = df_summary[df_summary['interpreter'] == interp_name]
        
        ## Generate shared coordinate twin system for score overlays
        ax_twin = ax.twinx()
        ax_twin.set_ylim(score_min - score_pad, score_max + score_pad)
        ax_twin.spines['right'].set_color('#555555')
        ax_twin.tick_params(axis='y', colors='#555555', labelsize=8)
        
        ## Plot distinct optimizer trajectories inside the current interpreter context
        unique_optimizers = interp_data['optimizer'].unique()
        colors = plt.cm.Set1(range(len(unique_optimizers)))
        
        for opt_idx, opt_name in enumerate(unique_optimizers):
            opt_data = interp_data[interp_data['optimizer'] == opt_name]
            color = colors[opt_idx]
            
            ### Render mean line trajectory for primary Normalized Hamming Distance
            h_line, = ax.plot(
                opt_data['iteration'], 
                opt_data['hamming_trajectory_mean'], 
                color=color, 
                lw=2, 
                linestyle='-'
            )
            ax.fill_between(
                opt_data['iteration'], 
                opt_data['hamming_trajectory_mean'] - opt_data['hamming_trajectory_sem'], 
                opt_data['hamming_trajectory_mean'] + opt_data['hamming_trajectory_sem'], 
                color=color, 
                alpha=0.12
            )
            
            ### Render corresponding consolidated score profile on the secondary vertical axis
            s_line, = ax_twin.plot(
                opt_data['iteration'], 
                opt_data[f"{score_col}_mean"], 
                color=color, 
                lw=1.5, 
                linestyle='--',
                alpha=0.85
            )
            ax_twin.fill_between(
                opt_data['iteration'], 
                opt_data[f"{score_col}_mean"] - opt_data[f"{score_col}_sem"], 
                opt_data[f"{score_col}_mean"] + opt_data[f"{score_col}_sem"], 
                color=color, 
                alpha=0.03
            )
            
            ### Cache handles for the unified global legend assembly
            if f"{opt_name} ($D_H$)" not in global_handles:
                global_handles[f"{opt_name} ($D_H$)"] = h_line
            if f"{opt_name} (Score)" not in global_handles:
                global_handles[f"{opt_name} (Score)"] = s_line
            
        ## Format axes styling and mathematical labels
        ax.set_title(f"Interpreter: {interp_name}", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Iteration ($t$)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if idx == 0:
            ax.set_ylabel("Normalized Hamming Distance ($D_H(t)$)", fontsize=10, fontweight='bold')
            
        if idx == num_plots - 1:
            ax_twin.set_ylabel("Consolidated Ensemble Score", fontsize=10, fontweight='bold', color='#333333', labelpad=10)
        else:
            ax_twin.set_yticklabels([])

    # Centralized layout processing
    ## Render single global horizontal legend beneath the subplots canvas
    fig.legend(
        list(global_handles.values()), 
        list(global_handles.keys()), 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.06), 
        ncol=len(global_handles) // 2, 
        frameon=True,
        fontsize=8.5
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Save framework output
    if save_path:
        logger.info("Saving consolidated Hamming panel report to: %s", save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()