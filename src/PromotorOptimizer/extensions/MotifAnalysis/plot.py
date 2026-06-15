import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_sequence(seq_id, sequence, df):

    fig, ax = plt.subplots(figsize=(14, 3.2))

    length = len(sequence)

    # backbone
    ax.hlines(0, 0, length, linewidth=2, color="black", alpha=0.7)

    subset = df[df["seq_id"] == seq_id].copy()

    if subset.empty:
        ax.set_title(f"{seq_id} (no motifs found)")
        ax.set_xlabel("Position")
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        return

    # assign each TF to a fixed track (VERY IMPORTANT CHANGE)
    unique_tfs = sorted(subset["TF_name"].unique())
    tf_to_track = {tf: i for i, tf in enumerate(unique_tfs)}

    # colors per TF
    cmap = plt.get_cmap("tab20")
    color_map = {tf: cmap(i % 20) for i, tf in enumerate(unique_tfs)}

    # normalize score for thickness
    scores = subset["score"]
    if scores.max() == scores.min():
        norm_scores = np.ones(len(scores))
    else:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

    # plot motifs in separate tracks
    for i, (_, row) in enumerate(subset.iterrows()):

        start = row["start"] - 1
        end = row["end"]
        tf = row["TF_name"]

        track_y = -tf_to_track[tf] * 0.15  # fixed lane per TF

        ax.plot(
            [start, end],
            [track_y, track_y],
            linewidth=2 + 4 * norm_scores.iloc[i],
            color=color_map.get(tf, "gray"),
            alpha=0.9,
        )

    # clean legend (ONLY place TF names)
    handles = [
        mpatches.Patch(color=color_map[tf], label=tf)
        for tf in unique_tfs[:20]
    ]

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title="TFs",
    )

    ax.set_title(f"Motif landscape: {seq_id}")
    ax.set_xlabel("Position (bp)")
    ax.set_yticks([])
    ax.set_xlim(0, length)

    # remove cluttered frame
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()