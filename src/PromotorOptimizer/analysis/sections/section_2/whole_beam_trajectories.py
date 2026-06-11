import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data processing and population statistics aggregation
## Calculate metric distributions across all target sequences and validator metrics

def compute_population_metrics(df):
    """
    Compute population-wide trajectories with mean and variance for Hamming distance,
    Shannon entropy, and validator-specific model architectures.

    :param df: Input optimization logs containing multi-sequence trajectories.
    :type df: pd.DataFrame
    :return: Processed long-form DataFrame with population metrics.
    :rtype: pd.DataFrame
    """
    # Population tracking metrics initialization
    ## List container for tracking structural divergence and information content
    population_records = []
    
    # Stratification of optimization groups
    ## Group dataset by distinct experimental setups to isolate variables
    grouped = df.groupby(['sequence_name', 'interpreter', 'optimizer'])
    
    for (seq_id, interp, opt), trajectory in grouped:
        ## Align trajectory checkpoints sequentially
        trajectory = trajectory.sort_values('iteration')
        
        ## Locate baseline genomic reference for the specific sequence ID
        baseline_df = trajectory[trajectory['iteration'] == 0]
        if baseline_df.empty:
            continue
        baseline_seq = baseline_df.iloc[0]['current_sequence']
        seq_len = len(baseline_seq)
        
        # Iteration-wise tracking loop
        ## Compute individual metrics for each iteration state
        for _, row in trajectory.iterrows():
            it = row['iteration']
            curr_seq = row['current_sequence']
            
            # Distance computation step
            ## Calculate normalized Hamming distance against absolute baseline state
            mismatches = sum(1 for b, c in zip(baseline_seq, curr_seq) if b != c)
            norm_hamming = mismatches / seq_len
            
            # Spatial focus computation step
            ## Calculate weight distribution entropy to identify local plateaus
            raw_w = np.array(row['interpreter_weights'])
            abs_w = np.abs(raw_w)
            w_sum = np.sum(abs_w)
            
            if w_sum > 0:
                ### Normalize array to extract absolute probability states
                prob_dist = abs_w / w_sum
                shannon_entropy_val = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
            else:
                ### Assign baseline value if sensitivity matrix flattens to absolute zero
                shannon_entropy_val = 0.0
                
            # Dictionary schema configuration
            ## Capture physical metrics alongside heterogeneous validator model predictions
            population_records.append({
                'sequence_name': seq_id,
                'interpreter': interp,
                'optimizer': opt,
                'iteration': it,
                'hamming_distance': norm_hamming,
                'shannon_entropy': shannon_entropy_val,
                'score_Deepstar': row['score_Deepstar'],
                'score_Second_Deepstarr': row['score_Second_Deepstarr'],
                'score_Model_Original': row['score_Model_Original']
            })
            
    # Matrix construction
    ## Aggregate individual dictionary entities into an immutable DataFrame
    return pd.DataFrame(population_records)


# Visualization orchestration for Analysis II.1 and Analysis II.2
## Multi-panel plotting functions mapping genotypic exploration and information focus

def generate_population_trajectory_plots(df_processed, save_prefix=None):
    """
    Generate population-level subplot figures tracking metrics with variance bounds
    and cross-model validator architectural drift.

    :param df_processed: Computed metric DataFrame from compute_population_metrics.
    :type df_processed: pd.DataFrame
    :param save_prefix: File prefix string used to save figures to local directory.
    :type save_prefix: str, optional
    """
    # -------------------------------------------------------------------------
    # ANALYSIS II.1: Genotypic Exploration Panels (Hamming Distance vs Validators)
    # -------------------------------------------------------------------------
    # Group optimization tracking metrics by interpreter facets
    unique_interpreters = df_processed['interpreter'].unique()
    num_interps = len(unique_interpreters)
    
    # Initialize composite exploration figure layout
    fig1, axes1 = plt.subplots(2, num_interps, figsize=(5 * num_interps, 9), sharex=True)
    if num_interps == 1:
        axes1 = np.expand_dims(axes1, axis=1)
        
    for idx, interp_name in enumerate(unique_interpreters):
        ## Extract subset corresponding to current interpreter configuration
        interp_mask = df_processed['interpreter'] == interp_name
        interp_data = df_processed[interp_mask]
        
        ## Top Row: Normalized Hamming Distance population distributions
        ax_ham = axes1[0, idx]
        for opt_name in interp_data['optimizer'].unique():
            opt_data = interp_data[interp_data['optimizer'] == opt_name]
            stats = opt_data.groupby('iteration')['hamming_distance'].agg(['mean', 'std']).reset_index()
            
            ### Render structural divergence average trajectory
            ax_ham.plot(stats['iteration'], stats['mean'], label=f"{opt_name}", lw=2)
            ### Overlay population variance envelope
            ax_ham.fill_between(stats['iteration'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], alpha=0.15)
            
        ax_ham.set_title(f"Interpreter: {interp_name}\nGenotypic Exploration", fontsize=11, fontweight='bold')
        if idx == 0:
            ax_ham.set_ylabel("Population Mean $D_H(t) \pm \sigma$", fontsize=10)
        ax_ham.grid(True, linestyle='--', alpha=0.5)
        ax_ham.legend(fontsize=8)
        
        ## Bottom Row: Validator multi-model trajectory comparison
        ax_val = axes1[1, idx]
        validators = ['score_Deepstar', 'score_Second_Deepstarr', 'score_Model_Original']
        for val_model in validators:
            #TODO: bugfixing
            # ### Compute aggregated mean performance over the sequence population pool
            # val_stats = interp_data.groupby('iteration')[val_model].mean().reset_index()
            # ax_val.plot(val_stats['iteration'], val_stats['mean'], label=val_model.replace('score_', ''), linestyle='-.')

            # Compute aggregated mean performance over the sequence population pool
            ## Perform groupby and aggregation
            val_stats = interp_data.groupby('iteration')[val_model].mean().reset_index()

            # Rename the aggregated column to 'mean' for consistent indexing
            ## Ensure the plotting call maps to the standardized name
            val_stats.rename(columns={val_model: 'mean'}, inplace=True)

            # Execute plot call
            ## Utilize the standardized 'mean' column index
            ax_val.plot(val_stats['iteration'], val_stats['mean'], label=val_model.replace('score_', ''), linestyle='-.')
            
        if idx == 0:
            ax_val.set_ylabel("Mean Validation Prediction Space", fontsize=10)
        ax_val.set_xlabel("Iteration ($t$)", fontsize=10)
        ax_val.grid(True, linestyle='--', alpha=0.5)
        ax_val.legend(fontsize=8)
        
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_analysis_II_1_hamming.png", dpi=300)
    plt.show()

    # -------------------------------------------------------------------------
    # ANALYSIS II.2: Attribution Focus Panels (Shannon Entropy vs Saturation)
    # -------------------------------------------------------------------------
    # Group validation tracking metrics by optimizer configurations
    unique_optimizers = df_processed['optimizer'].unique()
    num_opts = len(unique_optimizers)
    
    # Initialize composite entropy landscape figure layout
    fig2, axes2 = plt.subplots(2, num_opts, figsize=(5 * num_opts, 9), sharex=True)
    if num_opts == 1:
        axes2 = np.expand_dims(axes2, axis=1)
        
    for idx, opt_name in enumerate(unique_optimizers):
        ## Extract subset corresponding to current optimizer tracking loop
        opt_mask = df_processed['optimizer'] == opt_name
        opt_data = df_processed[opt_mask]
        
        ## Top Row: Shannon Entropy attention distribution
        ax_ent = axes2[0, idx]
        for interp_name in opt_data['interpreter'].unique():
            interp_data = opt_data[opt_data['interpreter'] == interp_name]
            stats = interp_data.groupby('iteration')['shannon_entropy'].agg(['mean', 'std']).reset_index()
            
            ### Render mean information landscape trajectory
            ax_ent.plot(stats['iteration'], stats['mean'], label=f"{interp_name}", lw=2)
            ### Overlay standard population variation boundaries
            ax_ent.fill_between(stats['iteration'], stats['mean'] - stats['std'], stats['mean'] + stats['std'], alpha=0.15)
            
        ax_ent.set_title(f"Optimizer: {opt_name}\nAttribution Entropy", fontsize=11, fontweight='bold')
        if idx == 0:
            ax_ent.set_ylabel("Population Mean $H(t) \pm \sigma$", fontsize=10)
        ax_ent.grid(True, linestyle='--', alpha=0.5)
        ax_ent.legend(fontsize=8)
        
        ## Bottom Row: Validation saturation velocity vs entropy flattening
        ## Bottom Row: Saturation trajectories across ALL validator architectures
        ax_sat = axes2[1, idx]
        validators = ['score_Deepstar', 'score_Second_Deepstarr', 'score_Model_Original']
        line_styles = ['-', '--', '-.']
        
        # Iterate through unique interpreter configurations
        for interp_name in opt_data['interpreter'].unique():
            interp_data = opt_data[opt_data['interpreter'] == interp_name]
            
            # Iterate through validator models to plot saturation trends
            for val_idx, val_model in enumerate(validators):
                ### Compute aggregated mean performance across the entire population pool
                score_stats = interp_data.groupby('iteration')[val_model].mean().reset_index()
                
                ### Define plot label and styling
                label_name = f"{interp_name} ({val_model.replace('score_', '')})"
                
                ### Plot trajectory using the explicit column name instead of 'mean'
                ax_sat.plot(score_stats['iteration'], score_stats[val_model], label=label_name, linestyle=line_styles[val_idx])
                
        if idx == 0:
            ax_sat.set_ylabel("Multi-Validator Prediction Space", fontsize=10)
        ax_sat.set_xlabel("Iteration ($t$)", fontsize=10)
        ax_sat.grid(True, linestyle='--', alpha=0.5)
        ax_sat.legend(fontsize=7, loc="lower right")
        
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_analysis_II_2_entropy.png", dpi=300)
    plt.show()