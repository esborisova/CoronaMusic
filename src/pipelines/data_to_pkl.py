"""Pipeline for saving plit tweets (music/not music) into single/two separate df(s)"""
import sys

sys.path.append("../scripts")
import ndjson
from itertools import chain
import pandas as pd
from collect_paths import save_filepaths

root_dir = [
    "../../Twitter/GeoCov19/data/music_related/",
    "../../Twitter/GeoCov19/data/not_music_related/",
]

files_paths = save_filepaths(root_dir)

music_tweets = []
not_music_tweets = []

for file in files_paths:
    with open(file) as f:
        try:
            data = ndjson.load(f)
        except ValueError:
            continue
        if len(data) != 0:
            if "/music_related/" in file:
                music_tweets.append(data)
            else:
                not_music_tweets.append(data)

df_music = pd.DataFrame(list(chain.from_iterable(music_tweets)))
df_not_music = pd.DataFrame(list(chain.from_iterable(not_music_tweets)))

df_music["music"] = True
df_not_music["music"] = False

df_music.to_pickle("../../split_data/music.pkl")
df_not_music.to_pickle("../../split_data/not_music.pkl")

merged = df_music.append(df_not_music, ignore_index=True)
merged.to_pickle("../../split_data/geocovid.pkl")
