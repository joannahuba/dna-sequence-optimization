# ANALYIS 1.1 
import numpy as np
import pandas as pd
from collections import defaultdict

#TODO przejrzeć i czy to działa poprawnei 

# Adaptive pattern discovery engine
## Dynamically adjusted window lengths and density thresholds based on trace entropy

def extract_adaptive_de_novo_motifs(df, target_seq="seq_4_no_active", min_k=6, max_k=14):
    """
    Perform de novo motif discovery using dynamic k-mer window sizing and an 
    adaptive weight threshold calculated from iteration-specific entropy states.

    :param df: Processed DataFrame containing metrics and raw interpreter weights.
    :type df: pd.DataFrame
    :param target_seq: Sequence identifier pool to isolate.
    :type target_seq: str
    :param min_k: Minimum bound for the sliding structural window.
    :type min_k: int
    :param max_k: Maximum bound for the sliding structural window.
    :type max_k: int
    :return: Dictionary containing the single most robust dynamic consensus motif footprint.
    :rtype: dict
    """
    sub_df = df[df['sequence_name'] == target_seq].copy()
    if sub_df.empty:
        return {}
        
    max_iteration = sub_df['iteration'].max()
    terminal_population = sub_df[sub_df['iteration'] == max_iteration]
    
    overall_best_kmer = None
    max_density_score = -1
    best_pfm = None
    
    # Grid search over spatial scales
    ## Dynamically evaluate pattern density across biological window lengths
    for kmer_len in range(min_k, max_k + 1):
        kmer_counts = defaultdict(int)
        fragments_pool = []
        
        for _, row in terminal_population.iterrows():
            sequence_string = row['current_sequence']
            weights_array = np.abs(np.array(row['interpreter_weights']))
            
            # Adaptive threshold formulation
            ## Derive quantile dynamically: lower entropy forces a more stringent cutoff
            normalized_w = weights_array / np.sum(weights_array)
            shannon_h = -np.sum(normalized_w * np.log2(normalized_w + 1e-12))
            max_h = np.log2(len(weights_array))
            
            ### Scaling function map: maps low entropy to high quantile, high entropy to low quantile
            adaptive_quantile = 0.50 + 0.45 * (1.0 - (shannon_h / max_h))
            cutoff_threshold = np.quantile(weights_array, adaptive_quantile)
            
            ## Scan sequence matrix positions
            for i in range(len(sequence_string) - kmer_len + 1):
                window_seq = sequence_string[i:i + kmer_len]
                window_w = weights_array[i:i + kmer_len]
                
                if np.mean(window_w) >= cutoff_threshold:
                    kmer_counts[window_seq] += 1
                    fragments_pool.append(window_seq)
                    
        # Performance validation step
        ## Evaluate consensus robust metrics for the current spatial window scale
        if kmer_counts:
            top_kmer, top_count = max(kmer_counts.items(), key=lambda x: x[1])
            if top_count > max_density_score:
                max_density_score = top_count
                overall_best_kmer = top_kmer
                
                ### Construct optimized position frequency matrix for selected length
                nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
                pfm = np.zeros((4, kmer_len))
                for frag in fragments_pool:
                    if len(frag) == kmer_len:
                        for pos, nuc in enumerate(frag):
                            if nuc in nucleotide_map:
                                pfm[nucleotide_map[nuc], pos] += 1
                best_pfm = pfm / (len(fragments_pool) + 1e-12)
                
    return {
        'optimal_dynamic_kmer': overall_best_kmer,
        'observed_frequency': max_density_score,
        'adaptive_position_frequency_matrix': best_pfm.tolist() if best_pfm is not None else []
    }