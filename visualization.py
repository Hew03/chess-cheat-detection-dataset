import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path

data_file = "data/game_features_with_cheating.parquet"
elo_ratings_file = "data/elo_ratings.pkl"
elo_bins_file = "data/elo_bins.pkl"
plots_dir = "plots"

Path(plots_dir).mkdir(exist_ok=True)

df = pd.read_parquet(data_file)

if os.path.exists(elo_bins_file):
    with open(elo_bins_file, "rb") as f:
        elo_bins = pickle.load(f)
else:
    with open(elo_ratings_file, "rb") as f:
        elo_ratings = pickle.load(f)
    num_bins = 10
    elo_bins = pd.qcut(elo_ratings, q=num_bins, duplicates="drop").categories
    with open(elo_bins_file, "wb") as f:
        pickle.dump(elo_bins, f)

elo_bin_labels = [f"{int(b.left)}-{int(b.right)}" for b in elo_bins]

bin_stats = df.groupby("elo_bin").agg({
    "avg_cpl": ["mean"],
    "move_match_rate": ["mean"]
}).reset_index()
bin_stats.columns = ["elo_bin", "avg_cpl_mean", "move_match_rate_mean"]

cheating_proportions = df.groupby("elo_bin")["is_cheating"].mean().reset_index()

bin_stats = bin_stats.set_index("elo_bin").reindex(elo_bin_labels).reset_index()
cheating_proportions = cheating_proportions.set_index("elo_bin").reindex(elo_bin_labels).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))
bar_width = 0.35
x = np.arange(len(elo_bin_labels))

ax1.bar(x - bar_width/2, bin_stats["avg_cpl_mean"], bar_width, label="Average CPL", color="skyblue")
ax1.set_xlabel("Elo Bin")
ax1.set_ylabel("Average CPL (centipawns)", color="skyblue")
ax1.tick_params(axis="y", labelcolor="skyblue")
ax1.set_xticks(x)
ax1.set_xticklabels(elo_bin_labels, rotation=45)

ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, bin_stats["move_match_rate_mean"], bar_width, label="Move Match Rate", color="salmon")
ax2.set_ylabel("Move Match Rate", color="salmon")
ax2.tick_params(axis="y", labelcolor="salmon")

fig.suptitle("Average CPL and Move Match Rate by Elo Bin")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "elo_bin_metrics.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(elo_bin_labels, cheating_proportions["is_cheating"], color="orange")
plt.xlabel("Elo Bin")
plt.ylabel("Proportion Flagged as Cheating")
plt.title("Proportion of Flagged Cheating Games by Elo Bin")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "cheating_proportions.png"))
plt.close()

sample_df = df.sample(n=min(1000, len(df)), random_state=42)
cheating = sample_df[sample_df["is_cheating"]]
non_cheating = sample_df[~sample_df["is_cheating"]]

plt.figure(figsize=(10, 6))
plt.scatter(non_cheating["avg_cpl"], non_cheating["move_match_rate"], c="skyblue", label="Non-Cheating", alpha=0.6)
plt.scatter(cheating["avg_cpl"], cheating["move_match_rate"], c="salmon", label="Cheating", alpha=0.6)
plt.xlabel("Average CPL (centipawns)")
plt.ylabel("Move Match Rate")
plt.title("Avg CPL vs. Move Match Rate (Cheating Flagged)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "cpl_vs_match_rate.png"))
plt.close()

print(f"Visualizations saved in '{plots_dir}' directory:")
print("- elo_bin_metrics.png")
print("- cheating_proportions.png")
print("- cpl_vs_match_rate.png")