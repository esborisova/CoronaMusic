import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_parodies = pd.read_pickle("BERT_scores/BERT_corona_parodies.pkl")
df_originals = pd.read_pickle("BERT_scores/BERT_matched_originals.pkl")

parodies = df_parodies[str(sys.argv[1])].tolist()
parodies_df0 = pd.DataFrame(parodies)
parodies_df0["Tag"] = "Parodies"

originals = df_originals[str(sys.argv[1])].tolist()
originals_df0 = pd.DataFrame(originals)
originals_df0["Tag"] = "Originals"

conct_df = parodies_df0.append(originals_df0, ignore_index=True)

if "sent" in str(sys.argv[1]):
    value_vars = ["NEG", "NEU", "POS"]
    var_name = "Sentiment"
else:
    value_vars = ["others", "joy", "sadness", "anger", "surprise", "disgust", "fear"]
    var_name = "Emotions"

df0 = pd.melt(
    conct_df, value_vars=value_vars, id_vars="Tag", var_name=var_name, value_name="Mean"
)

ci = [95, "sd"]

for item in ci:
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=var_name, y="Mean", hue="Tag", ci=item, ax=ax, data=df0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[2:], labels=labels[2:])

    print(item)

    if item == 95:
        name = "_mean_ci95.png"
    else:
        name = "_mean_sd.png"

    filename = var_name + name
    print(filename)
    fig.savefig(filename)
