import sys 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import datetime
import matplotlib.dates as md

def transform_date(date):
    new_date = datetime.strftime(datetime.strptime(date,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
    return new_date

df = pd.read_pickle(f"../../split_data/{str(sys.argv[1])}")
df["created_at"] = df["created_at"].apply(transform_date)

labels = ["NEG", "NEU", "POS"]

scores = df["BERT_sent_scores"].tolist()
dates = df["created_at"].tolist()
df0 = pd.DataFrame(scores)

df0["Date"] = dates
df0['Date'] = pd.to_datetime(df0['Date'], format = '%Y-%m-%d')
grouped_by_date = df0.groupby(['Date']).mean()
grouped_by_date = grouped_by_date.reset_index()

sns.set(font_scale=1.5)
fig, axs = plt.subplots(3, 1, figsize=(15, 15))

for ax, label, in zip(axs.ravel(), labels):
    sns.lineplot(x="Date", y=label, data = grouped_by_date, ax = ax)
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %d'))
    ax.set_xlabel(None)
    
if "not_music" in {str(sys.argv[1])}:
    title =  "BERT sentiment scores over time in non music"
else:
    title = "BERT sentiment scores over time in music"

sub_title = str(sys.argv[1]).split(".")[0]

fig.supxlabel("Date", fontsize = 25)
fig.supylabel("Sentiment", fontsize = 25)
plt.suptitle(title, fontsize = 20, y=0.92, fontweight = "bold")
title = title.replace(' ', '_')
fig.savefig(f"../../figs/{title}_{sub_title}.png")