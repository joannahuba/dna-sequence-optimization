# Heading 1 (Response Serialization Engine)
## Inject cross-validation predictions directly into the unified models_predictions dictionary map
import json
from typing import Dict, Any
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def update_json_with_predictions(
    json_path: str,
    output_path: str,
    predictions: Dict[str, Dict[str, float]]
) -> None:
    """
    Updates the modern optimization log structure by merging cross-model evaluation scores
    directly into the existing 'models_predictions' fields across trajectories and beams[cite: 24].

    :param json_path: Path to the original input results JSON log file.
    :type json_path: str
    :param output_path: Destination path where the enriched JSON data will be stored.
    :type output_path: str
    :param predictions: Look-up table mapping sequences to model scores: {seq: {model: score}}.
    :type predictions: Dict[str, Dict[str, float]]
    """
    with open(json_path, "r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    for seq_id, seq_payload in raw_data.items():
        if not isinstance(seq_payload, dict):
            continue

        optimizers_dict = seq_payload.get("optimizers", {})
        for opt_name, opt_payload in optimizers_dict.items():
            if not isinstance(opt_payload, dict):
                continue
            interpreters_dict = opt_payload.get("interpreters", {})
            for interp_name, interp_payload in interpreters_dict.items():
                if not isinstance(interp_payload, dict):
                    continue
                models_dict = interp_payload.get("models", {})
                for model_key, model_payload in models_dict.items():
                    if not isinstance(model_payload, dict):
                        continue
                    steps = model_payload.get("steps", [])
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                            
                        if "current_sequence" in step:
                            seq = step["current_sequence"]
                            if seq in predictions:
                                if "models_predictions" not in step:
                                    step["models_predictions"] = {}
                                step["models_predictions"].update(predictions[seq])
                        
                        # In-place beam population prediction updates
                        beam_pop = step.get("beam_population", [])
                        for beam_node in beam_pop:
                            if isinstance(beam_node, dict) and "sequence" in beam_node:
                                b_seq = beam_node["sequence"]
                                if b_seq in predictions:
                                    if "models_predictions" not in beam_node:
                                        beam_node["models_predictions"] = {}
                                    beam_node["models_predictions"].update(predictions[b_seq])

    with open(output_path, "w", encoding="utf-8") as output_handle:
        json.dump(raw_data, output_handle, indent=4, ensure_ascii=False)
    logger.info("Enriched unified dataset log successfully saved to: %s", output_path)