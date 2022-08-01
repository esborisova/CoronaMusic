import sys

import pandas as pd
import regex as re
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento import create_analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def clean_songs(text: str) -> str:

    cleaned = text.replace("\n", " ")
    cleaned = re.sub("[\[].*?[\]]", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    return cleaned


tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(
    "pysentimiento/robertuito-sentiment-analysis"
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

analyzer_sent = create_analyzer(task="sentiment", lang="en")
analyzer_em = create_analyzer(task="emotion", lang="en")

df = pd.read_pickle(str(sys.argv[1]))
df["prepared_BERT"] = df["raw_text"].apply(clean_songs)
texts = df["prepared_BERT"].tolist()

df["BERT_sent_scores"] = ""
df["BERT_emot_scores"] = ""

sentiment_predictions = analyzer_sent.predict(texts)
sentiment_scores = [prediction.probas for prediction in sentiment_predictions]
df["BERT_sent_scores"] = sentiment_scores

emotion_predictions = analyzer_em.predict(texts)
emotion_scores = [prediction.probas for prediction in emotion_predictions]
df["BERT_emot_scores"] = emotion_scores

if "orona_songs" in str(sys.argv[1]):
    name = "BERT_corona_songs.pkl"

elif "corona_parodies" in str(sys.argv[1]):
    name = "BERT_corona_parodies.pkl"
else:
    name = "BERT_matched_originals.pkl"

path = "BERT_scores/" + name
df.to_pickle(name)
