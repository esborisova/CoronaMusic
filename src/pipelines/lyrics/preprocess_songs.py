import sys


sys.path.append("../../../scripts")
import pandas as pd
import spacy
from preprocess import clean_text, collect_lemmas, collect_nn_adj, rm_stops

stops = open("stops.txt", "r")
stops = stops.read().split()
nlp = spacy.load("en_core_web_lg")

df = pd.read_pickle("BERT_scores/BERT_corona_songs.pkl")
tags = ["PROPN", "NOUN", "ADJ"]

texts = df["prepared_BERT"].tolist()
cleaned = [clean_text(text) for text in texts]
df["cleaned_text"] = cleaned
lemmas = [collect_lemmas(text, nlp) for text in cleaned]
nn_adj = [collect_nn_adj(text, nlp, tags) for text in cleaned]
no_stops_lemmas = [rm_stops(text, stops) for text in lemmas]
no_stops_nn_adj = [rm_stops(text, stops) for text in nn_adj]
df["lemmas"] = no_stops_lemmas
df["nn_adj"] = no_stops_nn_adj
# df = df.loc[df["lemmas"] != ""]

df.to_pickle("topic_model/corona_songs_preprocessed.pkl")
