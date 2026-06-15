import pandas as pd
import MOODS.scan
import MOODS.tools
from pyjaspar import jaspardb


class MOODSScanner:
    def __init__(
        self,
        threshold: float = 0.0001,
        species: int = 9606,
        collection: str = "CORE",
    ):
        self.threshold = threshold

        db = jaspardb(release="JASPAR2024")

        motifs = db.fetch_motifs(
            collection=collection,
            tax_group=None,
            species=[species]
        )

        self.motif_names = []
        self.motif_lengths = []
        matrices = []

        for motif in motifs:
            pwm = motif.pwm

            matrix = [
                pwm["A"],
                pwm["C"],
                pwm["G"],
                pwm["T"]
            ]

            bg = [0.25] * 4

            matrices.append(
                MOODS.tools.log_odds(
                    matrix,
                    bg,
                    0.001
                )
            )

            self.motif_names.append(motif.name)
            self.motif_lengths.append(len(matrix[0]))

        self.thresholds = [
            MOODS.tools.threshold_from_p(
                matrix,
                bg,
                self.threshold
            )
            for matrix in matrices
        ]

        self.scanner = MOODS.scan.Scanner(7)

        self.scanner.set_motifs(
            matrices,
            bg,
            self.thresholds
        )

    def scan(self, fasta: dict) -> pd.DataFrame:

        rows = []

        for seq_id, sequence in fasta.items():

            results = self.scanner.scan(sequence)

            for motif_idx, hits in enumerate(results):

                motif_name = self.motif_names[motif_idx]
                motif_len = self.motif_lengths[motif_idx]

                for hit in hits:

                    start = hit.pos + 1
                    end = start + motif_len - 1

                    rows.append(
                        {
                            "seq_id": seq_id,
                            "TF_name": motif_name,
                            "motif_name": motif_name,
                            "motif_seq": sequence[
                                start - 1:end
                            ],
                            "start": start,
                            "end": end,
                            "strand": "+",
                            "score": hit.score,
                        }
                    )

        return pd.DataFrame(rows)