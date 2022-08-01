import sys

sys.append.path("../../scripts")
import pandas as pd
from gensim.models import LdaModel
import pyLDAvis.gensim_models
import pyLDAvis.sklearn
from topic_model import prepare_input, compute_c_v


labels = ["lemmas", "nn_adj"]

for label in labels:
    df = pd.read_pickle("topic_model/corona_songs_preprocessed.pkl")
    texts = df[label].tolist()

    bigram, dictionary, corpus = prepare_input(texts)
    max_topics = 10
    min_topics = 2
    step = 2

    x = range(min_topics, max_topics, step)
    coherence_values = compute_c_v(
        dictionary=dictionary,
        corpus=corpus,
        texts=texts,
        min_topics=min_topics,
        max_topics=max_topics,
        step=step,
    )

    best_result_index = coherence_values.index(max(coherence_values))
    ldamodel = LdaModel(
        corpus=corpus,
        num_topics=x[best_result_index],
        id2word=dictionary,
        update_every=1,
        passes=10,
        per_word_topics=True,
    )

    model = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
    filename = label + "_corona_songs.html"
    pyLDAvis.save_html(model, filename)
