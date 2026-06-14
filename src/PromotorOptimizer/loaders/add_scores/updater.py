import json
from typing import Dict

# Response serialization engine
## Inject multi-model scoring matrices back into the primary trajectory logs
def update_json_with_predictions(
    json_path: str,
    output_path: str,
    predictions: Dict[str, Dict[str, float]]
) -> None:
    """
    Updates the original optimization log structure with cross-model evaluation scores.

    :param json_path: Path to the original input results JSON log file.
    :type json_path: str
    :param output_path: Destination path where the enriched JSON data will be stored.
    :type output_path: str
    :param predictions: Look-up table mapping sequences to model scores: {seq: {model: score}}.
    :type predictions: Dict[str, Dict[str, float]]
    :return: None
    :rtype: None
    """
    # File loading operations
    ## Load source trajectory data structure
    with open(json_path, "r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    # In-place sequence decoration matrix loop
    ## Traverse sequence entries
    for seq_id, interpreters_dict in raw_data.items():
        if not isinstance(interpreters_dict, dict):
            continue
            
        for interpreter_name, interpreter_content in interpreters_dict.items():
            if not isinstance(interpreter_content, dict):
                continue
                
            optimizers_group = interpreter_content.get("optimizers_results", {})
            if not isinstance(optimizers_group, dict):
                continue
                
            for optimizer_name, optimizer_results in optimizers_group.items():
                if not isinstance(optimizer_results, dict):
                    continue
                    
                trajectory = optimizer_results.get("trajectory", [])
                
                for frame in trajectory:
                    if isinstance(frame, dict) and "sequence" in frame:
                        seq = frame["sequence"]
                        if seq in predictions:
                            frame["cross_scores"] = predictions[seq]
                    
                    if isinstance(frame, dict) and "beam_population" in frame and isinstance(frame["beam_population"], list):
                        for beam_node in frame["beam_population"]:
                            if isinstance(beam_node, dict) and "sequence" in beam_node:
                                b_seq = beam_node["sequence"]
                                if b_seq in predictions:
                                    beam_node["cross_scores"] = predictions[b_seq]

    # Persistent output storage serialization
    ## Write updated structural mapping matrix to file
    with open(output_path, "w", encoding="utf-8") as output_handle:
        json.dump(raw_data, output_handle, indent=4, ensure_ascii=False)