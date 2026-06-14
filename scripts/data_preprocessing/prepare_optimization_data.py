# scripts/prepare_optimization_data.py

import os
from typing import Dict
import pandas as pd


def parse_fasta_with_metadata(fasta_path: str) -> Dict[str, str]:
    """
    Parses a standard genomic FASTA file into an identifier-to-sequence map.

    :param fasta_path: Local filesystem path to the target FASTA formatted file.
    :type fasta_path: str
    :return: Dictionary holding sequence strings indexed by their structural headers.
    :rtype: Dict[str, str]
    """
    sequences = {}

    # Sequential stream reader initialization
    with open(fasta_path, "r") as f:
        seq_id = None
        seq_lines = []

        for line in f:
            line = line.strip()

            if not line:
                continue

            # Route lines based on header condition markers
            if line.startswith(">"):
                ## Save previously compiled sequence accumulation track
                if seq_id is not None:
                    sequences[seq_id] = "".join(seq_lines)

                ## Reset track variables for the newborn sequence context
                seq_id = line[1:]
                seq_lines = []
            else:
                ## Accumulate continuous nucleotide sequence lines
                seq_lines.append(line)

        ## Flush residual buffer elements post final iteration step
        if seq_id is not None:
            sequences[seq_id] = "".join(seq_lines)

    return sequences


def build_optimization_tsv(
    fasta_path: str,
    output_path: str,
) -> None:
    """
    Assembles a synchronized sequence TSV file compatible with SCAN optimizers in maximization mode.

    :param fasta_path: Input source filepath containing FASTA sequence entities.
    :type fasta_path: str
    :param output_path: Destination coordinate path to export the parsed TSV matrix.
    :type output_path: str
    """
    # Import sequence tracks from local storage
    seqs = parse_fasta_with_metadata(fasta_path)
    rows = []

    # Processing loop
    for seq_id, seq in seqs.items():
        ## Construct rows strictly with structural identification and genomic strings
        rows.append({
            "id": seq_id,
            "sequence": seq
        })

    # Data export phase
    ## Initialize dataframe workspace and enforce output directory persistence
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ## Commit matrix array elements to an un-indexed dense TSV format
    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"[OK] Saved optimization input TSV -> {output_path}")


if __name__ == "__main__":
    # Execute transformation pipeline orchestration sweep for maximization mode
    build_optimization_tsv(
        fasta_path="data/subtaskB.fa",
        output_path="data/optimization_input.tsv"
    )