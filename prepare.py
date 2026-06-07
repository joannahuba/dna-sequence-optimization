from typing import Dict, Tuple
import pandas as pd


def parse_fasta_with_metadata(fasta_path: str) -> Dict[str, str]:
    """
    Reads FASTA like:
    >seq1_broken
    ACTG...
    """
    sequences = {}

    with open(fasta_path, "r") as f:
        seq_id = None
        seq_lines = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if seq_id is not None:
                    sequences[seq_id] = "".join(seq_lines)

                seq_id = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if seq_id is not None:
            sequences[seq_id] = "".join(seq_lines)

    return sequences


def build_reconstruction_tsv(
    fasta_path: str,
    output_path: str,
    mutation_map: Dict[str, int],
    activity_map: Dict[str, float],
):
    """
    mutation_map:
        seq_id -> number of introduced mutations

    activity_map:
        seq_id -> original activity
    """

    seqs = parse_fasta_with_metadata(fasta_path)

    rows = []

    for seq_id, seq in seqs.items():

        if seq_id not in mutation_map:
            raise ValueError(f"Missing mutation info for {seq_id}")

        if seq_id not in activity_map:
            raise ValueError(f"Missing activity info for {seq_id}")

        rows.append({
            "id": seq_id,
            "sequence": seq,
            "introduced_mutations": int(mutation_map[seq_id]),
            "original_activity": float(activity_map[seq_id]),
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"[OK] Saved reconstruction TSV -> {output_path}")


if __name__ == "__main__":

    mutation_map = {
        "seq_1_broken": 40,
        "seq_2_broken": 16,
        "seq_3_broken": 20,
    }

    activity_map = {
        "seq_1_broken": 2.5,
        "seq_2_broken": 2.7,
        "seq_3_broken": 1.1,
    }

    build_reconstruction_tsv(
        fasta_path="subtaskA.fa",
        output_path="data/reconstruction_input.tsv",
        mutation_map=mutation_map,
        activity_map=activity_map
    )