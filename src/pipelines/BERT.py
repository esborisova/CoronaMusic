import sys

sys.path.append("../scripts")
from clean_tweets import clean_text
import pandas as pd
import os
import regex as re
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento import create_analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

root_dir = "../../split_data/"
file_path = os.path.join(root_dir, str(sys.argv[1]))
df = pd.read_pickle(file_path)
df = df.filter(["id", "created_at", "from_user_id", "text"])

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

df["cleaned"] = ""
df["BERT_sent"] = ""
df["BERT_emot"] = ""

df["cleaned"] = df["text"].apply(clean_text)
tweets = df["cleaned"].tolist()
df["cleaned"] = preprocess_tweet(tweets, lang="en")

sentiment_predictions = analyzer_sent.predict(tweets)
sentiment_tags = [prediction.output for prediction in sentiment_predictions]
df["BERT_sent"] = sentiment_tags

emotion_predictions = analyzer_em.predict(tweets)
emotion_tags = [prediction.output for prediction in emotion_predictions]
df["BERT_emot"] = emotion_tags

df.to_pickle(f"../../split_data/BERT_{str(sys.argv[1])}")
df0 = df.drop_duplicates(subset=["text"], keep="first")
df0.to_pickle(f"../../split_data/BERT_{str(sys.argv[1])}_no_rts.pkl")
