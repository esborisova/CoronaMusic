import sys

sys.path.append("../../scripts")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as md
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from prepare_date import transform_date, date_df, get_average_score


df_music = pd.read_pickle(f"../../../split_data/{str(sys.argv[1])}")
df_not_music = pd.read_pickle(f"../../../split_data/{str(sys.argv[1])}")

df_music["created_at"] = df_music["created_at"].apply(transform_date)
df_not_music["created_at"] = df_not_music["created_at"].apply(transform_date)

music = date_df(df_music, "BERT_emot_scores")
not_music = date_df(df_not_music, "BERT_emot_scores")

average_music = get_average_score(music)
average_music["Tweet group"] = "Music"

average_not_music = get_average_score(not_music)
average_not_music["Tweet group"] = "Not music"

merged = average_music.append(average_not_music, ignore_index=True)

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
    ax = sns.lineplot(x="Date", y=label, data=merged, ax=ax, hue="Tweet group")
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ax.set_xlabel(None)
    ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

fig.legend(handles, labels, bbox_to_anchor=(1.03, 0.885), fontsize=20)
ax7.set_xlabel("Date", labelpad=30, fontsize=25)
fig.supylabel("Emotion", y=0.5, x=0.04, fontsize=25)

if "_no_rts_" in str(sys.argv[1]):
    label = "(without RTs)"
    label2 = "no_rts"
else:
    label = "(with RTs)"
    label2 = "with_rts"

plt.suptitle(
    f"Average daily BERT emotions: Music vs non-music {label}", y=0.91, fontsize=20
)
fig.savefig(f"../../../figs/daily_emot_dist_{label2}.png", bbox_inches="tight")
