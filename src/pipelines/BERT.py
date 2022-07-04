import sys

sys.path.append("../scripts")
from clean_tweets import clean_text
import pandas as pd
import regex as re
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento import create_analyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

df = pd.read_pickle(str(sys.argv[1]))
df = df.filter(["id", "created_at", "from_user_id", "text"])

tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(
    "pysentimiento/robertuito-sentiment-analysis"
)

# torch.cuda_set_device(1)
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

df.to_pickle(f"../../split_data/BERT_{str(sys.argv[1])}.pkl")
