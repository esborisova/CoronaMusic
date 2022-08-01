import sys

sys.append.path("../src/scripts")
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.plotting import save
import pandas as pd
import itertools
import collections
from collections import Counter
from nltk.util import bigrams
from bigrams import scale, bigram_freq, plot_bigrams

df = pd.read_pickle("topic_model/corona_songs_preprocessed.pkl")

labels = ["lemmas", "nn_adj"]
color_palette_nodes = ["#E0FFFF", "#bcdfeb", "#62b4cf", "#1E90FF"]
color_palette_edges = ["#8fbc8f", "#3cb371", "#2e8b57", "#006400"]

for label in labels:
    lemmas = df[label].tolist()
    terms_bigram = [list(bigrams(tweet)) for tweet in lemmas]

    bigram = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigram)
    bigrams_df = pd.DataFrame(
        bigram_counts.most_common(30), columns=["bigram", "count"]
    )

    word_freq = Counter(lemma for tweet in lemmas for lemma in set(tweet))
    w_freq_df = pd.DataFrame(
        word_freq.items(), columns=["word", "frequency"]
    ).sort_values(by="frequency", ascending=False)

    freq_dict = w_freq_df.to_dict(orient="split")["data"]
    co_occurence = dict(bigrams_df.values)

    palette_nodes = ["#8fbc8f", "#3cb371", "#2e8b57", "#006400"]
    d = bigrams_df.set_index("bigram").T.to_dict("records")
    G = nx.Graph()

    for key, value in d[0].items():
        G.add_edge(key[0], key[1], weight=(value))

    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(G, k=4)

    scaled_co_occurence = scale(co_occurence)
    freq = bigram_freq(freq_dict, G, scale=True)

    d = w_freq_df.to_dict(orient="split")["data"]
    d = [(int(word[1])) * 2 for node in G.nodes() for word in d if word[0] == node]

    nx.draw_networkx(
        G,
        pos,
        font_size=10,
        width=3,
        cmap="Blues",
        edge_color=palette_nodes,
        node_color=d,
        with_labels=False,
        ax=ax,
    )

    for key, value in pos.items():
        x, y = value[0], value[1]
        ax.text(
            x,
            y,
            s=key,
            bbox=dict(facecolor="#FFF0F5", alpha=0.5, edgecolor="grey", pad=3.5),
            horizontalalignment="center",
            weight="bold",
            fontsize=6,
        )

    fig.patch.set_visible(False)
    ax.axis("off")
    filename = label + "_bigrams_coronasongs.png"
    plt.savefig(filename, dpi=150)

    fig = plot_bigrams(
        G,
        freq,
        scaled_co_occurence,
        pos,
        color_palette_nodes,
        color_palette_edges,
        "Bigrams",
    )
    name = label + "_bigrams_coronasongs.html"
    save(fig, filename=name)
