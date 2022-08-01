import sys


sys.path.append("../../scripts")
import pandas as pd
from collect_paths import save_filepaths


files_paths = save_filepaths(str(sys.argv[1]))

data = []

for path in files_paths:

    f = open(path, errors="ignore", encoding="utf-8")
    text = f.read()
    data.append(text)


df = pd.DataFrame(data, columns=["raw_text"])

if "1_all_corona_songs/" in str(sys.argv[1]):
    name = "corona_songs.pkl"

elif "2a_corona_parodies/" in str(sys.argv[1]):
    name = "corona_parodies.pkl"
else:
    name = "matched_originals.pkl"
path = "raw_data/" + name

df.to_pickle(path)
