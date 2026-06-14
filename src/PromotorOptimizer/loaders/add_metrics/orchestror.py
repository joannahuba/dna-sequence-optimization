import pandas as pd
from typing import List
from .base_metric import MetricCalculator
from .exampe import HammingTrajectoryCalculator, ShannonEntropyCalculator 

# metric
## All metrics registry
def get_all_available_metric():
    all_available_metric = [
        HammingTrajectoryCalculator(),
        ShannonEntropyCalculator()
    ]
    return all_available_metric

# Pipeline Orchestration Logic
## Core functions to execute registered metrics over optimization trajectories

def calculate_trajectory_metrics(
        df: pd.DataFrame, 
        calculators: List[MetricCalculator] = get_all_available_metric()
    ) -> pd.DataFrame:
    """
    Process optimization logs by executing a collection of metric calculators 
    over isolated sequence trajectories.

    :param df: The input DataFrame containing sequence optimization trajectories.
    :type df: pd.DataFrame
    :param calculators: A list of instantiated MetricCalculator subclasses to execute.
    :type calculators: List[MetricCalculator]
    :return: A DataFrame containing the original trajectory keys along with all computed metrics.
    :rtype: pd.DataFrame
    """
    # Initialize results container
    ## List to accumulate row-by-row dictionary records
    metric_records = []
    
    # Trajectory grouping
    ## Group sequences by unique identifier, interpreter, and optimizer combinations
    trajectory_groups = df.groupby(['sequence_name', 'interpreter', 'optimizer'])
    
    for (seq_id, interpreter_name, optimizer_name), trajectory in trajectory_groups:
        ## Ensure sequential alignment of iterations
        trajectory = trajectory.sort_values('iteration')
        
        # Metric state initialization
        ## Allow each calculator to pre-compute constants for the current trajectory group
        for calculator in calculators:
            calculator.prepare(trajectory)
            
        # Iteration-wise metric parsing
        ## Loop through each sequential step in the trajectory
        for _, row in trajectory.iterrows():
            ### Core identifiers representing the unique state point
            record = {
                'sequence_name': seq_id,
                'interpreter': interpreter_name,
                'optimizer': optimizer_name,
                'iteration': row['iteration']
            }
            
            ### Dynamically execute and merge outputs from all registered calculators
            for calculator in calculators:
                metric_output = calculator.calculate(row)
                record.update(metric_output)
                
            metric_records.append(record)
            
    # Return compiled structure
    ## Construct a standard DataFrame from records
    return pd.DataFrame(metric_records)