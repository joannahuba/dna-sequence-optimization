from Bio import SeqIO


def load_fasta(fasta_path: str) -> dict:
    return {
        record.id: str(record.seq)
        for record in SeqIO.parse(fasta_path, "fasta")
    }