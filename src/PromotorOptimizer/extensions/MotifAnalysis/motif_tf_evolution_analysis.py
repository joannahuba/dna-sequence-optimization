import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

from .scanner import MOODSScanner


# ============================================================
# 1. INTERNAL SCANNER WRAPPER (DF → MOODS scan)
# ============================================================

def _scan_df_sequences(df, scanner):
    """
    Runs MOODS scan on DF-based sequences.

    Returns expanded motif-hit dataframe.
    """

    rows = []

    for _, row in df.iterrows():

        seq_id = row["sequence_name"]
        sequence = row["current_sequence"]
        iteration = row["iteration"]

        results = scanner.scanner.scan(sequence)

        for motif_idx, hits in enumerate(results):

            motif_name = scanner.motif_names[motif_idx]
            motif_len = scanner.motif_lengths[motif_idx]

            for hit in hits:

                start = hit.pos + 1
                end = start + motif_len - 1

                rows.append({
                    "sequence_name": seq_id,
                    "iteration": iteration,
                    "motif_name": motif_name,
                    "start": start,
                    "end": end,
                    "score": hit.score
                })

    return pd.DataFrame(rows)


# ============================================================
# 2. TOP-N MOTIFS PER ITERATION
# ============================================================

def motif_evolution_from_df(
    df,
    top_n=5,
    sequence_filter=None
):
    """
    Runs full MOODS scan directly on DF and tracks top-N motifs.
    """

    scanner = MOODSScanner()

    scan_df = _scan_df_sequences(df, scanner)

    if scan_df.empty:
        raise ValueError("No motif hits found")

    results = []

    for it in sorted(scan_df["iteration"].unique()):

        iter_df = scan_df[scan_df["iteration"] == it]

        counts = iter_df["motif_name"].value_counts()

        top = counts.head(top_n)

        for rank, (motif, freq) in enumerate(top.items(), start=1):
            results.append({
                "iteration": it,
                "rank": rank,
                "motif": motif,
                "frequency": freq
            })

    return pd.DataFrame(results), scan_df


# ============================================================
# 3. STATISTICS
# ============================================================

def motif_stats_from_df(scan_df):
    """
    Diversity + dominance per iteration.
    """

    stats = []

    for it in sorted(scan_df["iteration"].unique()):

        iter_df = scan_df[scan_df["iteration"] == it]

        counts = iter_df["motif_name"].value_counts()

        if len(counts) == 0:
            continue

        stats.append({
            "iteration": it,
            "dominance": counts.iloc[0] / counts.sum(),
            "diversity": len(counts),
            "total_hits": counts.sum()
        })

    return pd.DataFrame(stats)


# ============================================================
# 4. PLOT
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# stable stretch transform
# ---------------------------
def stretch_y(y, pivot=160):
    """
    Expands 0–pivot region and compresses values above pivot.
    Keeps continuity (no jumps).
    """

    y = np.array(y, dtype=float)

    out = np.zeros_like(y)

    low = y <= pivot
    high = ~low

    # expand low region
    out[low] = y[low] * 2.0

    # compress high region smoothly continuing from pivot
    out[high] = (pivot * 2.0) + (y[high] - pivot) * 0.35

    return out


def plot_motif_evolution(topn_df):

    fig, ax = plt.subplots(figsize=(12, 10))

    for motif in topn_df["motif"].unique():

        sub = topn_df[topn_df["motif"] == motif]

        ax.plot(
            sub["iteration"],
            stretch_y(sub["frequency"], pivot=160),
            marker="o",
            label=motif
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Motif frequency (stretched scale)")
    ax.set_title("Motif evolution (nonlinear Y stretching)")

    # ---------------------------
    # readable ticks (IMPORTANT FIX)
    # ---------------------------
    yticks_raw = np.array([0, 50, 100, 150, 160, 200, 260, 400, 550, 670])
    ax.set_yticks(stretch_y(yticks_raw, pivot=160))
    ax.set_yticklabels(yticks_raw)

    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()