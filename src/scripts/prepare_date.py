from datetime import datetime
import pandas as pd


def transform_date(date):
    new_date = datetime.strftime(
        datetime.strptime(date, "%a %b %d %H:%M:%S +0000 %Y"), "%Y-%m-%d"
    )
    return new_date


def date_df(df, score):
    bert_scores = df[score].tolist()
    dates = df["created_at"].tolist()
    df0 = pd.DataFrame(bert_scores)
    df0["Date"] = dates
    df0["Date"] = pd.to_datetime(df0["Date"], format="%Y-%m-%d")
    return df0


def get_average_score(df):
    grouped = df.groupby(["Date"]).mean()
    grouped = grouped.reset_index()
    return grouped


def get_week_numb(date):
    week = datetime.strftime(datetime.strptime(date,'%Y-%m-%d'), '%V')
    return week