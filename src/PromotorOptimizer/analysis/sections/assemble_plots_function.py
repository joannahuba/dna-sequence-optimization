import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Callable
from ...utils.logger import get_custom_logger

# Logging infrastructure initialization
logger = get_custom_logger(__name__)

# Assembly orchestration engine
## High-level data routing and layout supervisor with global legending and file-saving

def assemble_trajectory_report(
    df: pd.DataFrame,
    seq_list_reconstructed: List[Tuple[str, float, int]],
    seq_list_optimized: List[str],
    optimizers: List[str],
    interpreters: List[str],
    models: List[str],
    plot_engine: Callable[[plt.Axes, pd.DataFrame, float, Dict[str, Any]], None],
    plot_config: Dict[str, Any] = None,
    output_dir: str = None
) -> Dict[str, plt.Figure]:
    """
    Execute high-level dataset parsing loops and assemble structured multi-panel figures
    with a single unified global figure legend and automated disk export capabilities.

    :param df: Input DataFrame containing global optimization logging tracks.
    :type df: pd.DataFrame
    :param seq_list_reconstructed: Collection of sequence metadata triplets for reconstruction pipelines.
    :type seq_list_reconstructed: List[Tuple[str, float, int]]
    :param seq_list_optimized: Collection of sequence names targeting unconstrained optimization.
    :type seq_list_optimized: List[str]
    :param optimizers: List of optimization strategy keys defining matrix columns.
    :type optimizers: List[str]
    :param interpreters: List of attribution interpreter keys defining matrix rows.
    :type interpreters: List[str]
    :param models: List of core model architectures requiring independent figures.
    :type models: List[str]
    :param plot_engine: Functional reference responsible for low-level cell rendering.
    :type plot_engine: Callable[[plt.Axes, pd.DataFrame, float, Dict[str, Any]], None]
    :param plot_config: Supplementary parameters forwarded down to the cell plot engine, defaults to None.
    :type plot_config: Dict[str, Any], optional
    :param output_dir: Destination folder path where figures will be saved. If None, saving is skipped. Defaults to None.
    :type output_dir: str, optional
    :return: Map of structural identifier string keys to assembled plt.Figure instances.
    :rtype: Dict[str, plt.Figure]
    """
    generated_report = {}
    num_rows = len(interpreters)
    num_cols = len(optimizers)

    # Parsing block for target reconstructions
    ## Loop over distinct biological lineages and structural model conditions
    for seq_name, target_expr, _ in seq_list_reconstructed:
        for model_type in models:
            logger.info("Assembling matrix layout grid for sequence: %s | Model: %s", seq_name, model_type)
            
            fig, axes = plt.subplots(
                nrows=num_rows, 
                ncols=num_cols, 
                figsize=(num_cols * 4.5, num_rows * 3.8), 
                sharex=True, 
                sharey=True
            )
            
            ### Standardize dimensional indexing shapes for small matrices
            if num_rows == 1 and num_cols == 1:
                axes = np.array([[axes]])
            elif num_rows == 1:
                axes = axes[np.newaxis, :]
            elif num_cols == 1:
                axes = axes[:, np.newaxis]

            has_plotted_nodes = False
            legend_captured = False
            global_handles, global_labels = [], []

            ### Traversal over structural layout matrices
            for r_idx, interpreter in enumerate(interpreters):
                for c_idx, optimizer in enumerate(optimizers):
                    ax1 = axes[r_idx, c_idx]
                    
                    #### Isolate running path elements matching index definitions
                    mask = (
                        (df['sequence_name'] == seq_name) & 
                        (df['interpreter'] == interpreter) & 
                        (df['optimizer'] == optimizer) &
                        (df['model_type'] == model_type)
                    )
                    sub_df = df[mask].copy()
                    
                    if sub_df.empty:
                        ax1.text(0.5, 0.5, "Empty Path", ha='center', va='center', alpha=0.3, fontsize=9)
                        continue
                        
                    has_plotted_nodes = True
                    plot_engine(ax1, sub_df, target_expr, plot_config)
                    
                    #### Capture line handles from the first valid cell to build the global legend
                    if not legend_captured:
                        h1, l1 = ax1.get_legend_handles_labels()
                        global_handles.extend(h1)
                        global_labels.extend(l1)
                        ##### Check for twinx axes elements matching this cell's layout context
                        for ax_obj in fig.axes:
                            if ax_obj != ax1 and hasattr(ax_obj, 'get_subplotspec') and ax_obj.get_subplotspec() == ax1.get_subplotspec():
                                h2, l2 = ax_obj.get_legend_handles_labels()
                                global_handles.extend(h2)
                                global_labels.extend(l2)
                        legend_captured = True
                    
                    #### Orchestrate contextual outer matrix header decorations
                    if r_idx == 0:
                        ax1.set_title(optimizer, fontsize=10, fontweight='bold', pad=10)
                    if c_idx == 0:
                        ax1.set_ylabel(f"{interpreter}\n\nValidator Scale", fontsize=9, fontweight='bold')
                    if r_idx == num_rows - 1:
                        ax1.set_xlabel("Iterations", fontsize=9)

            ### Package populated canvas parameters into reference repository
            if has_plotted_nodes:
                dict_key = f"GRID_RECONSTRUCT_{seq_name}_{model_type}"
                fig.suptitle(f"Reconstruction Matrix Grid | Sequence: {seq_name} | Model: {model_type}", fontsize=13, fontweight='bold', y=0.98)
                
                #### Render single centralized global legend at the bottom of the figure layout
                if legend_captured:
                    fig.legend(global_handles, global_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(global_labels), frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
                
                fig.tight_layout(rect=[0, 0.04, 1, 0.95])
                generated_report[dict_key] = fig
                
                #### File storage block execution
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    export_path = os.path.join(output_dir, f"{dict_key}.png")
                    fig.savefig(export_path, dpi=300, bbox_inches='tight')
                    logger.info("Saved composite grid report to: %s", export_path)
            else:
                plt.close(fig)

    # Parsing block for unconstrained optimization loops
    ## Replicate geometric grid routing without applying target threshold baselines
    for seq_name in seq_list_optimized:
        for model_type in models:
            logger.info("Assembling matrix layout grid for optimized tracking: %s | Model: %s", seq_name, model_type)
            
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 4.5, num_rows * 3.8), sharex=True, sharey=True)
            
            if num_rows == 1 and num_cols == 1:
                axes = np.array([[axes]])
            elif num_rows == 1:
                axes = axes[np.newaxis, :]
            elif num_cols == 1:
                axes = axes[:, np.newaxis]

            has_plotted_nodes = False
            legend_captured = False
            global_handles, global_labels = [], []

            for r_idx, interpreter in enumerate(interpreters):
                for c_idx, optimizer in enumerate(optimizers):
                    ax1 = axes[r_idx, c_idx]
                    
                    mask = (
                        (df['sequence_name'] == seq_name) & 
                        (df['interpreter'] == interpreter) & 
                        (df['optimizer'] == optimizer) &
                        (df['model_type'] == model_type)
                    )
                    sub_df = df[mask].copy()
                    
                    if sub_df.empty:
                        ax1.text(0.5, 0.5, "Empty Path", ha='center', va='center', alpha=0.3, fontsize=9)
                        continue
                        
                    has_plotted_nodes = True
                    plot_engine(ax1, sub_df, None, plot_config)
                    
                    if not legend_captured:
                        h1, l1 = ax1.get_legend_handles_labels()
                        global_handles.extend(h1)
                        global_labels.extend(l1)
                        for ax_obj in fig.axes:
                            if ax_obj != ax1 and hasattr(ax_obj, 'get_subplotspec') and ax_obj.get_subplotspec() == ax1.get_subplotspec():
                                h2, l2 = ax_obj.get_legend_handles_labels()
                                global_handles.extend(h2)
                                global_labels.extend(l2)
                        legend_captured = True
                    
                    if r_idx == 0:
                        ax1.set_title(optimizer, fontsize=10, fontweight='bold', pad=10)
                    if c_idx == 0:
                        ax1.set_ylabel(f"{interpreter}\n\nValidator Scale", fontsize=9, fontweight='bold')
                    if r_idx == num_rows - 1:
                        ax1.set_xlabel("Iterations", fontsize=9)

            if has_plotted_nodes:
                dict_key = f"GRID_OPTIMIZED_{seq_name}_{model_type}"
                fig.suptitle(f"Optimized Profile Matrix Grid | Sequence: {seq_name} | Model: {model_type}", fontsize=13, fontweight='bold', y=0.98)
                
                if legend_captured:
                    fig.legend(global_handles, global_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=len(global_labels), frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
                
                fig.tight_layout(rect=[0, 0.04, 1, 0.95])
                generated_report[dict_key] = fig
                
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    export_path = os.path.join(output_dir, f"{dict_key}.png")
                    fig.savefig(export_path, dpi=300, bbox_inches='tight')
                    logger.info("Saved composite grid report to: %s", export_path)
            else:
                plt.close(fig)

    return generated_report