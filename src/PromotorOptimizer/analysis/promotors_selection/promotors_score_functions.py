import numpy as np
import pandas as pd



#TODO - moduł do przepisania, zamieszczam stare funkcje ponieważ posłużą za template do robienia df'ów

# Decision selection engine
## Rank and isolate top biological candidate sequences without external database lookups

def rank_sequences_by_bps(df, de_novo_kmers, target_seq="seq_4_no_active"):
    """
    Calculate the Biological Priority Score (BPS) to filter and select top
    optimized candidate sequences based on cross-model validation consensus
    and de novo motif density.

    :param df: Input DataFrame containing multi-model scores and sequences.
    :type df: pd.DataFrame
    :param de_novo_kmers: List of top conserved k-mer strings from de novo discovery.
    :type de_novo_kmers: list
    :param target_seq: Identifier of the sequence population pool to rank.
    :type target_seq: str
    :return: Formatted DataFrame ranked by BPS containing the top target variants.
    :rtype: pd.DataFrame
    """
    # Isolate targets and final iterations
    ## Focus prioritization on the mature optimization phase
    sub_df = df[df['sequence_name'] == target_seq].copy()
    if sub_df.empty:
        return pd.DataFrame()
        
    max_it = sub_df['iteration'].max()
    terminal_df = sub_df[sub_df['iteration'] == max_it].copy()
    
    # Min-Max Normalization of Validator Spaces
    ## Scale heterogeneous model outputs to unified [0, 1] intervals for cross-comparison
    validators = ['score_Deepstar', 'score_Second_Deepstarr', 'score_Model_Original']
    for val_model in validators:
        min_val = terminal_df[val_model].min()
        max_val = terminal_df[val_model].max()
        
        ### Prevent division by zero if scores are completely uniform
        if max_val - min_val > 0:
            terminal_df[f'norm_{val_model}'] = (terminal_df[val_model] - min_val) / (max_val - min_val)
        else:
            terminal_df[f'norm_{val_model}'] = 1.0
            
    # Consensus and motif density computation
    ## Evaluate sequence fitness metrics across independent architectures
    ranked_records = []
    for idx, row in terminal_df.iterrows():
        sequence = row['current_sequence']
        
        ### Compute cross-model orthogonality penalty via minimum operation
        norm_scores = [row[f'norm_{m}'] for m in validators]
        consensus_score = np.min(norm_scores)
        
        ### Count occurrences of identified population-wide de novo driving motifs
        motif_count = sum(sequence.count(kmer) for kmer in de_novo_kmers)
        
        ### Calculate final combined selection priority rank
        bps_value = consensus_score * (1 + motif_count)
        
        ranked_records.append({
            'file_id': row['file_id'],
            'current_sequence': sequence,
            'consensus_score': consensus_score,
            'motif_density': motif_count,
            'BPS': bps_value,
            'score_Deepstar': row['score_Deepstar']
        })
        
    # Output structure formatting
    ## Convert data to sorted execution rank order
    ranked_df = pd.DataFrame(ranked_records)
    if not ranked_df.empty:
        ranked_df = ranked_df.sort_values(by='BPS', ascending=False).reset_index(drop=True)
        
    return ranked_df


def extract_elite_sequences(df, target_optimization_column='ensemble_score'):
    """
    Group the trajectory pool and extract the exact row variants containing the maximum
    performance value for the specified evaluation metric.

    :param df: Input optimization trajectory DataFrame.
    :type df: pd.DataFrame
    :param target_optimization_column: The target score column used to define the absolute 'best' state.
    :type target_optimization_column: str
    :return: Selected elite rows containing structural sequences and corresponding validation scores.
    :rtype: pd.DataFrame
    """
    # Dynamic validator discovery
    ## Identify which model frameworks are embedded inside the current dataset
    validators = identify_validation_columns(df)
    
    # Required field validation
    ## Define core demographic identifiers needed for structural analysis
    required_metadata = ['sequence_name', 'interpreter', 'optimizer', 'iteration', 'current_sequence']
    output_columns = required_metadata + [target_optimization_column] + [v for v in validators if v != target_optimization_column]
    
    elite_records = []
    
    # Trajectory grouping execution
    ## Stratify the dataset by independent optimization runs
    grouped = df.groupby(['sequence_name', 'interpreter', 'optimizer'])
    
    for (seq_id, interp, opt), trajectory in grouped:
        ## Locate the row index containing the global maximum of the target metric
        if trajectory[target_optimization_column].dropna().empty:
            continue
            
        max_row_index = trajectory[target_optimization_column].idxmax()
        best_row = trajectory.loc[max_row_index]
        
        ## Construct clean dictionary record filtering out redundant operational variables
        record = {col: best_row[col] for col in output_columns if col in best_row}
        elite_records.append(record)
        
    # Return compiled framework
    ## Convert the accumulated structural records back into an aggregated DataFrame
    return pd.DataFrame(elite_records)


import numpy as np
import pandas as pd

# Population consensus selection engine
## Profile matrix aggregation and representative sequence extraction

def select_by_population_consensus(df, target_seq, budget, top_quantile=0.15):
    """
    Extract the most representative sequence by constructing a position-probability
    matrix from top-performing variants and ranking them by profile alignment.

    :param df: Input DataFrame containing optimization trajectories.
    :type df: pd.DataFrame
    :param target_seq: Identifier of the sequence subtask (e.g., 'Seq 1').
    :type target_seq: str
    :param budget: Maximum allowed mutations from the iteration 0 baseline.
    :type budget: int
    :param top_quantile: Top fraction of high-scoring sequences used to build the profile.
    :type top_quantile: float
    :return: Dictionary containing the optimal consensus-aligned sequence data.
    :rtype: dict
    """
    # Filter sub-task population
    ## Isolate target sequences across all logged iterations
    sub_df = df[df['sequence_name'] == target_seq].copy()
    if sub_df.empty:
        return {}
        
    # Extract baseline reference
    ## Locate the original corrupted sequence at iteration 0
    baseline_row = sub_df[sub_df['iteration'] == 0]
    if baseline_row.empty:
        return {}
    baseline_seq = baseline_row.iloc[0]['current_sequence']
    seq_len = len(baseline_seq)
    
    # Pre-filter candidates by hard mutation budget
    ## Only evaluate rows that strictly respect the experimental constraints
    valid_rows = []
    for _, row in sub_df.iterrows():
        curr_seq = row['current_sequence']
        mutations = sum(1 for b, c in zip(baseline_seq, curr_seq) if b != c)
        
        if mutations <= budget:
            ### Calculate group consensus score across distinct architectures
            ensemble_fitness = np.min([row['score_Deepstar'], row['score_Second_Deepstarr'], row['score_Model_Original']])
            valid_rows.append((row, ensemble_fitness))
            
    if not valid_rows:
        return {}
        
    # Isolate elite sub-population
    ## Sort valid candidates to filter the top performance quantile layer
    valid_rows.sort(key=lambda x: x[1], reverse=True)
    cutoff_index = max(1, int(len(valid_rows) * top_quantile))
    elite_pool = valid_rows[:cutoff_index]
    
    # Construct Position Frequency Matrix (PFM)
    ## Map position-wise nucleotide choices across all elite trajectories
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    pfm = np.zeros((4, seq_len))
    
    for elite_row, _ in elite_pool:
        seq_str = elite_row['current_sequence']
        for pos, nuc in enumerate(seq_str):
            if nuc in nuc_to_idx:
                pfm[nuc_to_idx[nuc], pos] += 1
                
    ## Normalize frequencies to obtain probabilities
    ppm = pfm / len(elite_pool)
    
    # Rank candidates by Consensus Alignment Score (CAS)
    ## Identify the variant that shares the highest identity with the population profile
    best_variant = None
    max_cas = -1.0
    
    for elite_row, fitness in elite_pool:
        seq_str = elite_row['current_sequence']
        current_cas = 0.0
        
        for pos, nuc in enumerate(seq_str):
            if nuc in nuc_to_idx:
                ### Sum position-wise probabilities to evaluate global consensus alignment
                current_cas += ppm[nuc_to_idx[nuc], pos]
                
        current_cas /= seq_len
        
        if current_cas > max_cas:
            max_cas = current_cas
            best_variant = {
                'id': target_seq,
                'new_sequence': seq_str,
                'predicted_rna_dna_ratio': elite_row['score_Deepstar'],
                'ensemble_consensus_score': fitness,
                'consensus_alignment_score': current_cas,
                'mutations_used': sum(1 for b, c in zip(baseline_seq, seq_str) if b != c)
            }
            
    return best_variant