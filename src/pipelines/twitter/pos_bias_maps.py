import sys

sys.path.append("../../scripts")
import pandas as pd
import plotly.express as px
from prepare_date import transform_date, get_week_numb


def find_country(row):
    country_a2 = row.country_code
    data = location[location["alpha-2"] == country_a2]
    country_name = data.name.values
    if len(country_name) > 0:
        country_name = country_name[0]
    else:
        country_name = "Unknown"
    return country_name


colorscale = [
    [0, "rgba(214, 39, 40, 1)"],
    [0.5, "rgba(255, 255, 255, 0.85)"],
    [1, "rgba(71, 166, 50, 1)"],
]

location = pd.read_csv("../../../split_data/location/country_codes_all.csv")
location["alpha-2"] = location["alpha-2"].str.lower()

music_filepaths = [
    "../../../split_data/location/location_BERT_music.pkl",
    "../../../split_data/location/location_BERT_no_rts_music.pkl",
]
not_music_filepaths = [
    "../../../split_data/location/location_BERT_not_music.pkl",
    "../../../split_data/location/location_BERT_no_rts_notmusic.pkl",
]

for mus_path, notmus_path in zip(music_filepaths, not_music_filepaths):

    df_music = pd.read_pickle(mus_path)
    df_not_music = pd.read_pickle(notmus_path)

    music_sent_scores = df_music["BERT_sent_scores"].tolist()
    not_music_sent_scores = df_not_music["BERT_sent_scores"].tolist()

    music_sent_df = pd.DataFrame(music_sent_scores)
    music_sent_df = music_sent_df.rename(
        columns={"NEG": "NEG_music", "NEU": "NEU_music", "POS": "POS_music"}
    )

    not_music_sent_df = pd.DataFrame(not_music_sent_scores)
    not_music_sent_df = not_music_sent_df.rename(
        columns={"NEG": "NEG_not_music", "NEU": "NEU_not_music", "POS": "POS_not_music"}
    )

    music_bert = df_music.filter(items=["created_at", "country_code"])
    music = pd.concat([music_bert, music_sent_df], axis=1)

    not_music_bert = df_not_music.filter(items=["created_at", "country_code"])
    not_music = pd.concat([not_music_bert, not_music_sent_df], axis=1)

    music["created_at"] = music["created_at"].apply(transform_date)
    not_music["created_at"] = not_music["created_at"].apply(transform_date)

    music["week"] = music["created_at"].apply(get_week_numb)
    not_music["week"] = not_music["created_at"].apply(get_week_numb)

    music_mean = music.groupby(["week", "country_code"]).agg(
        {"NEG_music": "mean", "NEU_music": "mean", "POS_music": "mean"}
    )
    music_mean = music_mean.reset_index()

    not_music_mean = not_music.groupby(["week", "country_code"]).agg(
        {"NEG_not_music": "mean", "NEU_not_music": "mean", "POS_not_music": "mean"}
    )
    not_music_mean = not_music_mean.reset_index()

    music_mean["music_positivity_bias"] = music_mean.apply(
        lambda x: x.POS_music - x.NEG_music, axis=1
    )
    not_music_mean["not_music_positivity_bias"] = not_music_mean.apply(
        lambda x: x.POS_not_music - x.NEG_not_music, axis=1
    )
    merged = pd.merge(music_mean, not_music_mean)
    merged["country_name"] = merged.apply(find_country, axis=1)
    merged = merged[merged["country_name"] != "Unknown"]
    merged["positivity_bias"] = merged.apply(
        lambda x: x.music_positivity_bias - x.not_music_positivity_bias, axis=1
    )

    fig = px.choropleth(
        merged,
        locationmode="country names",
        locations="country_name",
        color="positivity_bias",
        hover_name="country_name",
        animation_frame="week",
        color_continuous_scale=colorscale,
        range_color=[-1, 1],
        height=600,
    )

    if "rts" in mus_path and notmus_path:
        label = "no_rts"
    else:
        label = "with_rts"

    fig.write_html(f"BERT_pos_bias_map_{label}.html")
