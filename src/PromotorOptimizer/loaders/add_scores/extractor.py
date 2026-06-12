import json
from typing import Dict, List, Tuple

# Ingestion engine
## Parse JSON structure to collect unique nucleotide sequences across trajectories
def extract_unique_sequences(json_path: str) -> Tuple[Dict, List[str]]:
    """
    Parses optimization log files to collect all unique DNA sequences.

    This function traverses both single-trajectory logs and deep-nested 
    population beam structures under the 'optimizers_results' key, extracting 
    unique sequences to prevent redundant forward-pass computations on the GPU.

    :param json_path: File system path to the target results JSON log.
    :type json_path: str
    :return: A tuple containing the raw loaded dictionary and a list of unique sequences.
    :rtype: Tuple[Dict, List[str]]
    """
    # File loading operations
    ## Execute safe file read stream
    with open(json_path, "r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    unique_sequences_set = set()

    # Dictionary tree traversal loop
    ## Iterate over sequence IDs present in the document root
    for seq_id, interpreters_dict in raw_data.items():
        if not isinstance(interpreters_dict, dict):
            continue
            
        ## Iterate over attribution method frameworks
        for interpreter_name, interpreter_content in interpreters_dict.items():
            if not isinstance(interpreter_content, dict):
                continue
                
            ### Access optimization execution metrics grouping
            optimizers_group = interpreter_content.get("optimizers_results", {})
            if not isinstance(optimizers_group, dict):
                continue
                
            ### Iterate over registered structural optimizers
            for optimizer_name, optimizer_results in optimizers_group.items():
                if not isinstance(optimizer_results, dict):
                    continue
                    
                trajectory = optimizer_results.get("trajectory", [])
                
                ### Extract sequences from chronological execution frames
                for frame in trajectory:
                    if isinstance(frame, dict) and "sequence" in frame:
                        unique_sequences_set.add(frame["sequence"])
                    
                    #### Capture deeply nested population variants from beam search matrices
                    if isinstance(frame, dict) and "beam_population" in frame and isinstance(frame["beam_population"], list):
                        for beam_node in frame["beam_population"]:
                            if isinstance(beam_node, dict) and "sequence" in beam_node:
                                unique_sequences_set.add(beam_node["sequence"])

    return raw_data, list(unique_sequences_set)