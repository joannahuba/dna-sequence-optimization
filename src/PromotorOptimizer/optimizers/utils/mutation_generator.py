# optimization/mutation_generator.py

import random
import numpy as np

BASES = ["A", "C", "G", "T"]


class MutationGenerator:

    @staticmethod
    def mutate_position(sequence, position, new_base):

        seq = list(sequence)
        seq[position] = new_base
        return "".join(seq)

    @staticmethod
    def random_mutation(sequence, n_mutations=1):

        seq = sequence

        positions = random.sample(
            range(len(sequence)),
            n_mutations
        )

        for pos in positions:
            current = seq[pos]
            candidates = [b for b in BASES if b != current]
            seq = MutationGenerator.mutate_position(
                seq,
                pos,
                random.choice(candidates)
            )

        return seq

    @staticmethod
    def guided_mutation(sequence, importance_scores, n_mutations=1, prefix_len=0, suffix_len=0):

        # importance_scores: (L, 4)
        probs = importance_scores.numpy().copy()
        ## fix cut adapters from mutated sequence: (if shape is reversed i reverse it again)
        ## Check if shape is (4, L) and transpose to standard (L, 4)
        if probs.shape[0] == 4 and probs.shape[1] == len(sequence):
            probs = probs.T

        # fix cut adapters from mutated sequence: Here i take beginning and end of sequence so it will not mutate
        ## Extract constant adapter blocks to preserve them from mutation
        prefix_seq = sequence[:prefix_len] if prefix_len > 0 else ""
        suffix_seq = sequence[len(sequence) - suffix_len:] if suffix_len > 0 else ""

        ## Slice core sequence and importance matrix to isolate the target promoter region
        end_idx = len(sequence) - suffix_len if suffix_len > 0 else len(sequence)
        core_seq = sequence[prefix_len:end_idx]
        
        prob_end_idx = probs.shape[0] - suffix_len if suffix_len > 0 else probs.shape[0]
        core_probs = probs[prefix_len:prob_end_idx, :]

        ## Ensure correct length alignment for the core region
        if core_probs.shape[0] != len(core_seq):
            core_probs = core_probs[:len(core_seq), :]
        
        ## Standardize score distribution values
        core_probs = np.abs(core_probs)
        core_probs = core_probs + 1e-8

        total_sum = core_probs.sum()
        if total_sum > 0:
            core_probs = core_probs / total_sum
        ## Handler: rarely but it can happen that we will obtain 0 from sum (to small representation)
        else:
            core_probs = np.ones_like(core_probs) / core_probs.size
        
        # fix cut adapters from mutated sequence: we doing it later)
        # DEPRECATED
        # seq = sequence

        ## Select positions and execute localized mutations within the core region
        position_probabilities = core_probs.sum(axis=1)
        positions = np.random.choice(
            len(core_seq),
            size=n_mutations,
            replace=False,
            p=position_probabilities
        )

        mutated_core = core_seq
        for pos in positions:

            base_probs = core_probs[pos]
            base_probs_sum = base_probs.sum()

            if base_probs_sum > 0:
                base_probs = base_probs / base_probs_sum
            else:
                base_probs = np.array([0.25, 0.25, 0.25, 0.25])

            new_base = np.random.choice(BASES, p=base_probs)

            mutated_core = MutationGenerator.mutate_position(
                mutated_core,
                int(pos),
                new_base
            )
        
        # fix cut adapters from mutated sequence: Here return full seqauences (data loader cuts it later)
        return prefix_seq + mutated_core + suffix_seq

    @staticmethod
    def hybrid_mutation(
        sequence,
        importance_scores,
        n_mutations,
        lambda_weight=0.7,
        # fix cut adapters from mutated sequence: get prefix and suffix for parameters
        prefix_len: int=0,
        suffix_len: int=0
    ):

        guided_n = int(n_mutations * lambda_weight)
        random_n = n_mutations - guided_n

        seq = MutationGenerator.guided_mutation(
            sequence,
            importance_scores,
            guided_n,
            # fix cut adapters from mutated sequence: pass prefix and suffix for parameters
            prefix_len=prefix_len,
            suffix_len=suffix_len
        )

        seq = MutationGenerator.random_mutation(
            seq,
            random_n
        )

        return seq
    
    @staticmethod
    def top_k_positions(importance_scores, k=20, reduction_mode="sum"):
        """
        Extracts top K most reactive sequence coordinates based on matrix reduction.

        :param importance_scores: Matrix containing sensitivity attribution scores.
        :type importance_scores: torch.Tensor or numpy.ndarray
        :param k: Footprint size of coordinate targets to isolate. Default is 20.
        :type k: int
        :param reduction: Reduction method flag ('sum' for gradient maps, 'max' for ISM). Default is 'sum'.
        :type reduction: str
        :return: Array containing sorted coordinate indices.
        :rtype: numpy.ndarray
        """
        # Handle PyTorch tensor conversion steps safely
        ## Isolate underlying array across compute device boundaries
        if hasattr(importance_scores, "detach"):
            tensor = importance_scores.clone().detach().cpu()
        else:
            import torch
            tensor = torch.tensor(importance_scores).cpu()

        # Execute dimensionality reduction profiles
        ## Select strategy based on interpretation model type
        if reduction_mode == "max":
            scores = tensor.max(dim=1)[0].numpy()
        else:
            scores = tensor.abs().sum(dim=1).numpy()

        return scores.argsort()[::-1][:k]