# Heading 1 (Ingestion Engine for Sequence Cross-Validation)
## Parse JSON structures to collect unique nucleotide sequences across multi-format trajectories
import json
from typing import Dict, List, Tuple, Any
from ...utils.logger import get_custom_logger

logger = get_custom_logger(__name__)


def extract_unique_sequences(json_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Parses both old and new optimization log structures to collect unique DNA sequences
    from both primary frames and deeply nested population beam matrices.

    :param json_path: File system path to the target results JSON log.
    :type json_path: str
    :return: A tuple containing the raw loaded dictionary and a list of unique sequences.
    :rtype: Tuple[Dict, List[str]]
    """
    with open(json_path, "r", encoding="utf-8") as file_handle:
        raw_data = json.load(file_handle)

    unique_sequences_set = set()

    for seq_id, seq_payload in raw_data.items():

        ## MODERN FORMAT TRAVERSAL
        optimizers_dict = seq_payload.get("optimizers", {})
        for opt_name, opt_payload in optimizers_dict.items():
            interpreters_dict = opt_payload.get("interpreters", {})
            for interp_name, interp_payload in interpreters_dict.items():
                models_dict = interp_payload.get("models", {})
                for model_key, model_payload in models_dict.items():
                    steps = model_payload.get("steps", [])
                    for step in steps:
                        if "current_sequence" in step:
                            unique_sequences_set.add(step["current_sequence"])
                        
                        ### Extract deeply nested population variants from beam matrices
                        beam_pop = step.get("beam_population", [])
                        for beam_node in beam_pop:
                            if isinstance(beam_node, dict) and "sequence" in beam_node:
                                unique_sequences_set.add(beam_node["sequence"])


    return raw_data, list(unique_sequences_set)