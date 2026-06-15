import subprocess
import pandas as pd


class JASPARMapper:
    """
    Correct biological mapping:
    MEME motifs → TOMTOM → JASPAR TFs
    """

    def __init__(self, jaspar_db: str = "JASPAR2024.meme"):
        self.jaspar_db = jaspar_db

    def run_tomtom(self, meme_file: str, out_dir: str = "tomtom_out"):
        cmd = [
            "tomtom",
            "-oc", out_dir,
            meme_file,
            self.jaspar_db
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        return f"{out_dir}/tomtom.tsv"

    def parse_tomtom(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", comment="#")

        # TOMTOM columns vary slightly but usually include:
        return df[[
            "Query_ID",
            "Target_ID",
            "q-value"
        ]].rename(columns={
            "Query_ID": "motif_name",
            "Target_ID": "TF_name",
            "q-value": "q_value"
        })

    def annotate(self, fimo_df: pd.DataFrame, tomtom_df: pd.DataFrame):
        """
        Map motif → TF using TOMTOM similarity
        """

        mapping = tomtom_df.groupby("motif_name")["TF_name"].apply(list).to_dict()

        fimo_df = fimo_df.copy()
        fimo_df["TF_name"] = fimo_df["motif_name"].map(
            lambda x: ",".join(mapping.get(x, ["Unknown_TF"]))
        )

        return fimo_df