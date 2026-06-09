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
    def guided_mutation(sequence, importance_scores, n_mutations=1):

        # importance_scores: (L, 4)

        probs = importance_scores.numpy().copy()
        probs = probs + 1e-8
        probs = probs / probs.sum()

        seq = sequence

        positions = np.random.choice(
            len(sequence),
            size=n_mutations,
            replace=False,
            p=probs.sum(axis=1) / probs.sum()
        )

        for pos in positions:

            base_probs = probs[pos]
            base_probs = base_probs / base_probs.sum()

            new_base = np.random.choice(BASES, p=base_probs)

            seq = MutationGenerator.mutate_position(
                seq,
                int(pos),
                new_base
            )

        return seq

    @staticmethod
    def hybrid_mutation(
        sequence,
        importance_scores,
        n_mutations,
        lambda_weight=0.7
    ):

        guided_n = int(n_mutations * lambda_weight)
        random_n = n_mutations - guided_n

        seq = MutationGenerator.guided_mutation(
            sequence,
            importance_scores,
            guided_n
        )

        seq = MutationGenerator.random_mutation(
            seq,
            random_n
        )

        return seq
    
    @staticmethod
    def top_k_positions(importance_scores, k=20):
        scores = importance_scores.sum(dim=1).detach().cpu().numpy()
        return scores.argsort()[::-1][:k]