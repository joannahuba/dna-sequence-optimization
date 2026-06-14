import matplotlib.pyplot as plt
import seaborn as sns

# Visualization Module
## Plotting functions for decoupling and trajectory analysis

def plot_shannon_entropy_panels(df_metrics, save_path=None):
    """
    Generate a multi-panel figure displaying Shannon Entropy trajectories across iterations.
    Each subplot represents an Optimizer, with lines labeled by Interpreters.

    :param df_metrics: DataFrame containing calculated metrics from Module 1.
    :type df_metrics: pd.DataFrame
    :param save_path: Optional file path to save the generated figure.
    :type save_path: str, optional
    """
    # Summary aggregation
    ## Compute mean and standard error across sequence replicates for clear visualization
    df_summary = df_metrics.groupby(['optimizer', 'interpreter', 'iteration'])['shannon_entropy'].agg(['mean', 'sem']).reset_index()
    
    # Grid initialization
    ## Identify unique optimizers to define the subplot layout dynamically
    unique_optimizers = df_summary['optimizer'].unique()
    num_plots = len(unique_optimizers)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)
    if num_plots == 1:
        axes = [axes]
        
    # Multi-panel execution
    ## Iterate over each optimizer to build localized axes
    for idx, opt_name in enumerate(unique_optimizers):
        ax = axes[idx]
        opt_data = df_summary[df_summary['optimizer'] == opt_name]
        
        ## Plot distinct interpreter trajectories inside the current optimizer context
        for interp_name in opt_data['interpreter'].unique():
            interp_data = opt_data[opt_data['interpreter'] == interp_name]
            
            ### Render mean line trajectory
            ax.plot(interp_data['iteration'], interp_data['mean'], label=interp_name, lw=2)
            ### Overlay standard error confidence boundaries
            ax.fill_between(
                interp_data['iteration'], 
                interp_data['mean'] - interp_data['sem'], 
                interp_data['mean'] + interp_data['sem'], 
                alpha=0.15
            )
            
        ## Format axes styling and mathematical labels
        ax.set_title(f"Optimizer: {opt_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Iteration ($t$)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Shannon Entropy ($H(t)$)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Interpreter", loc="upper right")
        
    plt.tight_layout()
    
    # Save framework output
    ## Export figure if destination path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hamming_trajectory_panels(df_metrics, save_path=None):
    """
    Generate a multi-panel figure displaying Cumulative Normalized Hamming Distance across iterations.
    Each subplot represents an Interpreter, with lines labeled by Optimizers.

    :param df_metrics: DataFrame containing calculated metrics from Module 1.
    :type df_metrics: pd.DataFrame
    :param save_path: Optional file path to save the generated figure.
    :type save_path: str, optional
    """
    # Summary aggregation
    ## Compute mean and standard error across sequence replicates for clear visualization
    df_summary = df_metrics.groupby(['interpreter', 'optimizer', 'iteration'])['hamming_trajectory'].agg(['mean', 'sem']).reset_index()
    
    # Grid initialization
    ## Identify unique interpreters to define the subplot layout dynamically
    unique_interpreters = df_summary['interpreter'].unique()
    num_plots = len(unique_interpreters)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), sharey=True)
    if num_plots == 1:
        axes = [axes]
        
    # Multi-panel execution
    ## Iterate over each interpreter to build localized axes
    for idx, interp_name in enumerate(unique_interpreters):
        ax = axes[idx]
        interp_data = df_summary[df_summary['interpreter'] == interp_name]
        
        ## Plot distinct optimizer trajectories inside the current interpreter context
        for opt_name in interp_data['optimizer'].unique():
            opt_data = interp_data[interp_data['optimizer'] == opt_name]
            
            ### Render mean line trajectory
            ax.plot(opt_data['iteration'], opt_data['mean'], label=opt_name, lw=2)
            ### Overlay standard error confidence boundaries
            ax.fill_between(
                opt_data['iteration'], 
                opt_data['mean'] - opt_data['sem'], 
                opt_data['mean'] + opt_data['sem'], 
                alpha=0.15
            )
            
        ## Format axes styling and mathematical labels
        ax.set_title(f"Interpreter: {interp_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Iteration ($t$)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Normalized Hamming Distance ($D_H(t)$)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Optimizer", loc="lower right")
        
    plt.tight_layout()
    
    # Save framework output
    ## Export figure if destination path is specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()