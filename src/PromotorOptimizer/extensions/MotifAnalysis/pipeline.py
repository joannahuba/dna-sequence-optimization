from ...utils.io import load_fasta
from .scanner import MOODSScanner


class MotifPipeline:
    def __init__(
        self,
        threshold: float = 1e-4,
        species: int = 9606,
    ):
        self.scanner = MOODSScanner(
            threshold=threshold,
            species=species,
        )

    def run(self, fasta_file: str):

        fasta = load_fasta(fasta_file)

        df = self.scanner.scan(fasta)

        return df, fasta