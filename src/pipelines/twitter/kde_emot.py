import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as md
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec

df_music = pd.read_pickle("../../../split_data/BERT_no_rts_music.pkl")
df_not_music = pd.read_pickle("../../../split_data/BERT_no_rts_not_music.pkl")

music_scores = df_music["BERT_emot_scores"].tolist()
music_df0 = pd.DataFrame(music_scores)
music_df0["Tweet group"] = "Music"

non_music_scores = df_not_music["BERT_emot_scores"].tolist()
non_music_df0 = pd.DataFrame(non_music_scores)
non_music_df0["Tweet group"] = "Not music"

merged = music_df0.append(non_music_df0, ignore_index=True)

labels = ["joy", "sadness", "anger", "surprise", "disgust", "fear", "others"]

gs = gridspec.GridSpec(4, 2)

sns.set(font_scale=1.4)
fig = plt.figure(figsize=(20, 20))

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[1, 0])
ax4 = plt.subplot(gs[1, 1])
ax5 = plt.subplot(gs[2, 0])
ax6 = plt.subplot(gs[2, 1])
ax7 = plt.subplot(gs[3, :])

axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

for (
    ax,
    label,
) in zip(axs, labels):
    ax = sns.kdeplot(
        data=merged, x=label, ax=ax, hue="Tweet group", common_norm=False, cut=0
    )
    ax.set_ylabel(None)
    ax.get_legend().remove()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

fig.legend(["Not music", "Music"], bbox_to_anchor=(1.03, 0.885), fontsize=20)
fig.supxlabel("Sentiment", y=0.05, fontsize=25)
fig.supylabel("Density", x=0.05, fontsize=25)
plt.suptitle(
    "BERT emotions distribution: Music vs non-music (without RTs)", y=0.91, fontsize=20
)
fig.savefig("../../../figs/emot_dist_kde_no_rts.png", bbox_inches="tight")
