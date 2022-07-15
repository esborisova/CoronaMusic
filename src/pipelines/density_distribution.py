import sys 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle(f"../../split_data/{str(sys.argv[1])}")

column_names = ["BERT_sent_scores", "BERT_emot_scores"]

for name in column_names:
    scores = df[name].tolist()
    df0 = pd.DataFrame(scores)
    fig = sns.displot(df0,  kind="kde", multiple="stack", alpha=.3, linewidth=0.5)
    fig.set(ylabel = "Data Density", xlabel = "BERT Probability Score")
    sub_title = str(sys.argv[1]).split(".")[0]
    fig.savefig(f"../../figs/{name}_{sub_title}.png")