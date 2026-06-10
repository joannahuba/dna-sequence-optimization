import torch
import logging
from .base import BaseInterpreter
from ..core.types import InterpretationResult
from ..utils.preprocessing import encode_one

logger = logging.getLogger(__name__)

BASES = ["A", "C", "G", "T"]


class InSilicoMutagenesis(BaseInterpreter):

    def explain(self, model_manager, sequence, model_type="ensemble"):

        return self.explain_batch(model_manager, [sequence], model_type)[0]
        logger.info(f"[Mutagenesis] Start | seq_len={len(sequence)}")

        device = model_manager.get_device()
        models = model_manager.get_models()

        seq_len = len(sequence)

        importance_maps = []
        model_scores = {}

        for name, meta in models.items():

            model = meta["model"]

            base_tensor = torch.tensor(
                encode_one(sequence),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                _, ratio = model(base_tensor)
                base_score = ratio.mean().item()

            model_scores[name] = base_score

            logger.info(f"[Mutagenesis] model={name} base_score={base_score:.6f}")

            importance = torch.zeros(seq_len, 4, device=device)

            importance = torch.zeros(seq_len, 4, device=device)

            # Generation of all single-point variants
            ## Pre-allocate trajectory coordinate mappings to construct a single batch
            mutated_sequences = []
            mutation_mapping = []

            for pos in range(seq_len):
                current_base = sequence[pos]
                for base_idx, new_base in enumerate(BASES):
                    if new_base == current_base:
                        continue
                    
                    ### Construct mutated sequence character array
                    mutated_seq = list(sequence)
                    mutated_seq[pos] = new_base
                    mutated_str = "".join(mutated_seq)
                    
                    mutated_sequences.append(mutated_str)
                    mutation_mapping.append((pos, base_idx))

            # Vectorized batch evaluation pass
            ## Encode all structural variants into a single dense matrix block to drop allocation limits
            encoded_mutations = np.array([encode_one(seq) for seq in mutated_sequences])
            x_mut = torch.tensor(encoded_mutations, dtype=torch.float32, device=device)

            ## Execute parallelized inference step on GPU in a single forward pass
            with torch.no_grad():
                _, ratio_mut = model(x_mut)

            # Map scores back to position coordinates
            for idx, (pos, base_idx) in enumerate(mutation_mapping):
                ### Account for fluctuating output dimensionality profiles safely using scalar item extractions
                if ratio_mut.ndim == 0:
                    mut_score = ratio_mut.item()
                elif ratio_mut.ndim == 1:
                    mut_score = ratio_mut[idx].item()
                else:
                    mut_score = ratio_mut[idx].mean().item()

                importance[pos, base_idx] = abs(mut_score - base_score)

            importance_maps.append(importance)

        importance_maps = torch.stack(importance_maps)

        importance = importance_maps.mean(dim=0) if model_type == "ensemble" else importance_maps[0]

        logger.info("[Mutagenesis] Finished")

        return InterpretationResult(
            method_name="Mutagenesis",
            importance_scores=importance.cpu(),
            sequence=sequence,
            model_scores=model_scores,
            metadata={}
        )
    def explain_batch(self, model_manager, sequences: list, model_type="ensemble"):
        """
        Computes In Silico Mutagenesis attribution maps for a batch of sequences parallelly.

        :param model_manager: Unified evaluation suite manager coordination stack.
        :param sequences: List of DNA sequence strings currently in the beam pool.
        :type sequences: list of str
        :param model_type: Averaging methodology flag ('ensemble' or 'single').
        :type model_type: str
        :return: List of InterpretationResult instances matching the input batch order.
        :rtype: list
        """
        # Context initialization
        ## Extract processing targets and hardware specifications
        logger.info(f"[Mutagenesis] Start batch processing | sequences_count={len(sequences)}")
        device = model_manager.get_device()
        models = model_manager.get_models()
        batch_size = len(sequences)
        seq_len = len(sequences[0])

        per_model_maps = []
        model_scores = {name: [] for name in models.keys()}

        # Combinatorial variant space construction
        ## Generate all single-point modifications across the entire sequence batch
        mutated_sequences = []
        mutation_mapping = []

        for b_idx, sequence in enumerate(sequences):
            for pos in range(seq_len):
                current_base = sequence[pos]
                for base_idx, new_base in enumerate(BASES):
                    if new_base == current_base:
                        continue
                    
                    ### Construct physical mutated text string
                    mutated_seq = list(sequence)
                    mutated_seq[pos] = new_base
                    mutated_str = "".join(mutated_seq)
                    
                    mutated_sequences.append(mutated_str)
                    mutation_mapping.append((b_idx, pos, base_idx))

        # Vectorized array encoding
        ## Pack thousands of text sequences into a single dense matrix block
        import numpy as np
        from ..utils.preprocessing import encode_one

        encoded_mutations = np.array([encode_one(seq) for seq in mutated_sequences])
        x_mut = torch.tensor(encoded_mutations, dtype=torch.float32, device=device)

        encoded_bases = np.array([encode_one(seq) for seq in sequences])
        x_base = torch.tensor(encoded_bases, dtype=torch.float32, device=device)

        # Ensemble execution pass
        ## Evaluate mutations concurrently using safe sub-batched inference steps on GPU
        for name, meta in models.items():
            model = meta["model"]
            model.eval()

            ### Compute reference baseline profiles for the batch
            with torch.no_grad():
                _, ratio_base = model(x_base)

            ### Parse reference scores across fluctuating tensor dimensions
            base_scores_list = []
            for b_idx in range(batch_size):
                if ratio_base.ndim == 0:
                    base_val = ratio_base.item()
                elif ratio_base.ndim == 1:
                    base_val = ratio_base[b_idx].item()
                else:
                    base_val = ratio_base[b_idx].mean().item()
                base_scores_list.append(base_val)
                model_scores[name].append(base_val)

            ### Initialize batch importance storage tensors
            importance_batch = torch.zeros(batch_size, seq_len, 4, device=device)

            ### Process mutated sequence array in chunks to prevent VRAM OOM anomalies
            chunk_size = 512 * 2
            ratio_mut_list = []

            for chunk_idx in range(0, len(mutated_sequences), chunk_size):
                chunk_seqs = mutated_sequences[chunk_idx:chunk_idx + chunk_size]
                encoded_chunk = np.array([encode_one(seq) for seq in chunk_seqs])
                x_chunk = torch.tensor(encoded_chunk, dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    _, ratio_chunk = model(x_chunk)
                
                #### Move chunk predictions back to host memory to keep VRAM footprint minimal
                if ratio_chunk.ndim == 0:
                    ratio_chunk = ratio_chunk.unsqueeze(0)
                ratio_mut_list.append(ratio_chunk.cpu())

            ### Concatenate all sub-batch outputs back into a single prediction tracking vector
            ratio_mut = torch.cat(ratio_mut_list, dim=0)

            ### Distribute deltas back to spatial coordinates using coordinate mappings
            for idx, (b_idx, pos, base_idx) in enumerate(mutation_mapping):
                if ratio_mut.ndim == 1:
                    mut_score = ratio_mut[idx].item()
                else:
                    mut_score = ratio_mut[idx].mean().item()

                importance_batch[b_idx, pos, base_idx] = abs(mut_score - base_scores_list[b_idx])

            per_model_maps.append(importance_batch)

        # Dimension reduction filtering
        ## Stack individual model prediction maps into a single unified tensor layout
        per_model_maps = torch.stack(per_model_maps) # Shape: (M, B, L, 4)

        if model_type == "ensemble":
            ### Compute the mean across the model ensemble axis
            final_importance_batch = per_model_maps.mean(dim=0)
        else:
            ### Extract the primary model map directly
            final_importance_batch = per_model_maps[0]

        # Structure compilation and output routing
        results = []
        for b_idx in range(batch_size):
            single_scores = {name: model_scores[name][b_idx] for name in models.keys()}
            results.append(
                InterpretationResult(
                    method_name="Mutagenesis",
                    importance_scores=final_importance_batch[b_idx].cpu(),
                    sequence=sequences[b_idx],
                    model_scores=single_scores,
                    metadata={}
                )
            )

        logger.info("[Mutagenesis] Finished batch execution successfully")
        return results