import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def compute_motif_coverage(sequence_length, df_subset):
    """
    Union-based coverage of sequence by motifs.
    """
    if df_subset.empty:
        return 0.0

    intervals = df_subset[["start", "end"]].values.tolist()
    intervals = sorted(intervals, key=lambda x: x[0])

    merged = []

    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    covered = sum(end - start + 1 for start, end in merged)

    return 100.0 * covered / sequence_length


def plot_sequence(seq_id, sequence, df):

    fig, ax = plt.subplots(figsize=(14, 3.5))

    length = len(sequence)

    # backbone
    ax.hlines(0, 0, length, linewidth=2, color="black", alpha=0.7)

    subset = df[df["seq_id"] == seq_id].copy()

    # ---------------------------
    # coverage metric
    # ---------------------------
    coverage = compute_motif_coverage(length, subset)

    ax.set_title(
        f"Motif landscape: {seq_id} | coverage = {coverage:.2f}%"
    )

    ax.set_xlabel("Position (bp)")
    ax.set_yticks([])

    ax.set_xlim(0, length)

    if subset.empty:
        plt.tight_layout()
        plt.show()
        return

    # assign TF tracks
    unique_tfs = sorted(subset["TF_name"].unique())
    tf_to_track = {tf: i for i, tf in enumerate(unique_tfs)}

    cmap = plt.get_cmap("tab20")
    color_map = {tf: cmap(i % 20) for i, tf in enumerate(unique_tfs)}

    # normalize score safely
    scores = subset["score"].values
    if scores.max() == scores.min():
        norm_scores = np.ones(len(scores))
    else:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

    # plot motifs
    for i, (_, row) in enumerate(subset.iterrows()):

        start = row["start"] - 1
        end = row["end"]
        tf = row["TF_name"]

        track_y = -tf_to_track[tf] * 0.15

        ax.plot(
            [start, end],
            [track_y, track_y],
            linewidth=2 + 4 * norm_scores[i],
            color=color_map.get(tf, "gray"),
            alpha=0.9,
        )

    # legend (top TFs only to avoid clutter)
    handles = [
        mpatches.Patch(color=color_map[tf], label=tf)
        for tf in unique_tfs[:20]
    ]

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title="TFs"
    )

    # clean frame
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()