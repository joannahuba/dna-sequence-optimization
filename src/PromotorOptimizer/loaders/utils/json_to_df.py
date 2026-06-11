import pandas as pd
import numpy as np
from typing import Dict, Any
import numpy

def aggregate_advanced_trajectory(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregates nested iteration trajectories, explicitly extracting metrics 
    from trajectory steps when they are structured as pandas DataFrames.
    Computes both rolling historical path variances and cross-sectional
    ensemble validation variances alongside their inverse mapping weights.

    :param data_dict: Nested dictionary returned by the folder parser.
    :type data_dict: Dict[str, Any]
    :return: Long-format DataFrame containing chronological step metrics.
    :rtype: pd.DataFrame
    """
    # Master tracking array for flattened records
    all_steps = []

    # Parse structural layout branches
    for file_id, interpreters in data_dict.items():
        ## Traverse sequence domains and interpreter spaces
        for sequence_name, optimizers in interpreters.items():
            for interpreter_name, interpreter_payload in optimizers.items():
                if not isinstance(interpreter_payload, dict) or "optimizers" not in interpreter_payload:
                    continue

                ## Traverse optimization algorithm execution paths
                for optimizer_name, optimizer_payload in interpreter_payload["optimizers"].items():
                    df_traj = optimizer_payload.get("trajectory", None)
                    
                    if df_traj is None or not isinstance(df_traj, pd.DataFrame) or df_traj.empty:
                        continue

                    ### Determine job type from structural identifiers
                    combined_str = f"{file_id}_{sequence_name}_{optimizer_name}".lower()
                    if "optimization" in combined_str:
                        job_type = "optimization"
                    elif "reconstruction" in combined_str:
                        job_type = "reconstruction"
                    else:
                        job_type = "unknown"

                    ### Extract interpreted model type from the keys
                    model_type = "unknown"
                    for model_key in ["_beam_mh", "_final_beam", "_beam_boltzman", "_mh"]:
                        if model_key in optimizer_name or model_key in sequence_name or model_key in file_id:
                            model_type = model_key.lstrip("_")
                            break

                    ### Track rolling historical baseline scores within this localized trajectory
                    history_scores = []

                    ### Extract dictionary structures row by row from the DataFrame
                    trajectory_records = df_traj.to_dict('records')

                    ### Iterate over individual chronological execution rows
                    for step in trajectory_records:
                        current_seq = step["sequence"] if "sequence" in step else ""
                        
                        if not current_seq or pd.isna(current_seq):
                            continue
                            
                        seq_len = len(current_seq)
                        gc_count = current_seq.upper().count("G") + current_seq.upper().count("C")
                        gc_content = gc_count / seq_len if seq_len > 0 else 0.0

                        #### Unpack and log current primary ensemble evaluation score
                        ensemble_score = float(step["score"]) if "score" in step else 0.0
                        history_scores.append(ensemble_score)

                        row = {
                            "file_id": file_id,
                            "sequence_name": sequence_name,
                            "interpreter": interpreter_name,
                            "optimizer": optimizer_name,
                            "job_type": job_type,
                            "model_type": model_type,
                            "iteration": int(step["iteration"]) if "iteration" in step else 0,
                            "ensemble_score": ensemble_score,
                            "gc_content": gc_content,
                            "temperature": step["temperature"] if "temperature" in step else 0.0,
                            'current_sequence': current_seq,
                            'current_beam_population': step['beam_population'] if 'beam_population' in step else None,
                            "interpreter_weights": step["interpreter_weights"] if "interpreter_weights" in step else None
                        }

                        raw_cross_scores = step["cross_scores"] if "cross_scores" in step else {}
                        
                        if isinstance(raw_cross_scores, str):
                            try:
                                cross_scores = ast.literal_eval(raw_cross_scores)
                            except (ValueError, SyntaxError):
                                cross_scores = {}
                        elif isinstance(raw_cross_scores, dict):
                            cross_scores = raw_cross_scores
                        else:
                            cross_scores = {}

                        model_values = []
                        if cross_scores:
                            for m_name, m_score in cross_scores.items():
                                clean_col = f"score_{m_name.replace(' ', '_')}"
                                row[clean_col] = float(m_score)
                                model_values.append(float(m_score))

                        #### Calculate individual historical trajectory metrics (Process Noise)
                        if len(history_scores) > 1:
                            var_ind = float(np.var(history_scores, ddof=1))
                            weight_ind = 1.0 / (1.0 + var_ind)
                        else:
                            var_ind = 0.0
                            weight_ind = 1.0

                        #### Calculate aggregated cross-validation ensemble metrics (Model Noise)
                        if len(model_values) > 1:
                            var_agg = float(np.var(model_values, ddof=1))
                            weight_agg = 1.0 / (1.0 + var_agg)
                        else:
                            var_agg = 0.0
                            weight_agg = 1.0

                        #### Inject decoupled structural metrics into record
                        row["variance_individual"] = var_ind
                        row["weight_individual"] = weight_ind
                        row["variance_aggregated"] = var_agg
                        row["weight_aggregated"] = weight_agg
                        
                        all_steps.append(row)

    # Convert results array to master long-format matrix
    if not all_steps:
        return pd.DataFrame()

    trajectory_df = pd.DataFrame(all_steps)
    
    # Maintain strict chronological sequence order for line plot rendering
    trajectory_df.sort_values(
        by=["file_id", "sequence_name", "interpreter", "optimizer", "iteration"], 
        inplace=True, 
        ignore_index=True
    )
    
    return trajectory_df