import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as md
from matplotlib.ticker import FormatStrFormatter

df_music = pd.read_pickle("../../split_data/BERT_no_rts_music.pkl")
df_not_music = pd.read_pickle("../../split_data/BERT_no_rts_not_music.pkl")

music_scores = df_music["BERT_sent_scores"].tolist()
music_df0 = pd.DataFrame(music_scores)
music_df0["Tweet group"] = "Music"

non_music_scores = df_not_music["BERT_sent_scores"].tolist()
non_music_df0 = pd.DataFrame(non_music_scores)
non_music_df0["Tweet group"] = "Not music"

merged = music_df0.append(non_music_df0, ignore_index=True)
labels = ["NEG", "NEU", "POS"]

sns.set(font_scale=1.2)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for (
    ax,
    label,
) in zip(axs.ravel(), labels):
    ax = sns.kdeplot(
        data=merged, x=label, ax=ax, hue="Tweet group", common_norm=False, cut=0
    )
    ax.set_ylabel(None)
    ax.get_legend().remove()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

fig.legend(["Not music", "Music"], bbox_to_anchor=(1.06, 0.885), fontsize=12)
fig.supxlabel("Sentiment", y=0.05, fontsize=15)
fig.supylabel("Density", x=0.01, fontsize=15)
plt.suptitle(
    "BERT sentiment distribution: Music vs non-music (without RTs)", y=0.91, fontsize=15
)
fig.savefig("../../figs/sent_dist_kde_no_rts.png", bbox_inches="tight")
