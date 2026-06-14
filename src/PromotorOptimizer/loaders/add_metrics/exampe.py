
from .base_metric import MetricCalculator 
import pandas as pd
import numpy as np

class HammingTrajectoryCalculator(MetricCalculator):
    """
    Computes Cumulative Normalized Hamming Distance relative to the iteration 0 baseline.
    """

    def __init__(self):
        self.baseline_sequence = None
        self.sequence_length = None

    def prepare(self, trajectory: pd.DataFrame) -> None:
        ## Extract the baseline sequence state at iteration 0
        baseline_subset = trajectory[trajectory['iteration'] == 0]
        if not baseline_subset.empty:
            self.baseline_sequence = baseline_subset.iloc[0]['current_sequence']
            self.sequence_length = len(self.baseline_sequence)
        else:
            self.baseline_sequence = None
            self.sequence_length = None

    def calculate(self, row: pd.Series) -> dict:
        if self.baseline_sequence is None or self.sequence_length == 0:
            return {'hamming_trajectory': np.nan}
        
        ## Sum the positions where characters differ from the reference baseline
        current_sequence = row['current_sequence']
        mismatches = sum(1 for char_0, char_t in zip(self.baseline_sequence, current_sequence) if char_0 != char_t)
        return {'hamming_trajectory': mismatches / self.sequence_length}


class ShannonEntropyCalculator(MetricCalculator):
    """
    Computes Shannon entropy of absolute attribution weights.
    """

    def prepare(self, trajectory: pd.DataFrame) -> None:
        ## Stateless metric; no trajectory-level preparation required
        pass

    def calculate(self, row: pd.Series) -> dict:
        ## Convert weight lists from format to numpy arrays
        raw_weights = np.array(row['interpreter_weights'])
        absolute_weights = np.abs(raw_weights)
        weights_sum = np.sum(absolute_weights)
        
        if weights_sum > 0:
            ### Normalize into a probability distribution
            probability_distribution = absolute_weights / weights_sum
            ### Compute entropy avoiding log2 of zero via a tiny constant float addition
            shannon_entropy_val = -np.sum(probability_distribution * np.log2(probability_distribution + 1e-12))
        else:
            ### Handle uniform or unassigned weight landscapes
            shannon_entropy_val = 0.0
            
        return {'shannon_entropy': shannon_entropy_val}