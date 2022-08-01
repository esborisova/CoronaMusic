import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as md
from matplotlib.ticker import FormatStrFormatter
from prepare_date import transform_date, date_df, get_average_score


df_music = pd.read_pickle(f"../../split_data/{str(sys.argv[1])}")
df_not_music = pd.read_pickle(f"../../split_data/{str(sys.argv[1])}")

df_music["created_at"] = df_music["created_at"].apply(transform_date)
df_not_music["created_at"] = df_not_music["created_at"].apply(transform_date)

music = date_df(df_music, "BERT_sent_scores")
not_music = date_df(df_not_music, "BERT_sent_scores")

average_music = get_average_score(music)
average_music["Tweet group"] = "Music"

average_not_music = get_average_score(not_music)
average_not_music["Tweet group"] = "Not music"

merged = average_music.append(average_not_music, ignore_index=True)

labels = ["NEG", "NEU", "POS"]

sns.set(font_scale=1.2)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for (
    ax,
    label,
) in zip(axs.ravel(), labels):
    sns.lineplot(x="Date", y=label, data=merged, ax=ax, hue="Tweet group")
    ax.xaxis.set_major_formatter(md.DateFormatter("%b %d"))
    ax.set_xlabel(None)
    ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

fig.legend(handles, labels, bbox_to_anchor=(1.06, 0.885), fontsize=12)
axs[2].set_xlabel("Date", labelpad=24, fontsize=20)
fig.supylabel("Sentiment", x=-0.01, fontsize=20)

if "_no_rts_" in str(sys.argv[1]):
    label = "(without RTs)"
    label2 = "no_rts"
else:
    label = "(with RTs)"
    label2 = "with_rts"

plt.suptitle(
    f"Average daily BERT sentiment: Music vs non-music {label}", y=0.91, fontsize=15
)
fig.savefig(f"../../figs/daily_sent_dist_{label2}.png", bbox_inches="tight")
