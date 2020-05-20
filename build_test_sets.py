import pandas as pd
import csv
from datetime import datetime

ES_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"


def get_date(d):
    return datetime.strptime(d, ES_DATE_FORMAT)


tweets = pd.read_csv("/home/bmazoyer/Dev/news_twitter/data/event2018_ann_url.tsv", sep="\t",
                     quoting=csv.QUOTE_ALL, dtype={"id": int, "label": float,	"created_at": str,
                                                   "text": str,	"url": str,	"hashtag": str})
tweets = tweets.sort_values("id").reset_index(drop=True)

news = pd.read_csv("/home/bmazoyer/Dev/news_twitter/data/event2018_news_url.tsv", sep="\t",
                     quoting=csv.QUOTE_ALL, dtype={"id": int, "label": float,	"created_at": str,
                                                   "text": str,	"url": str,	"hashtag": str, "title": str})
news["date"] = news["created_at"].apply(get_date)
news = news.sort_values("date").reset_index(drop=True)

window = int(tweets.shape[0]/4)
for i in range(0, tweets.shape[0], window):
    short = tweets[i:i+window]
    max_date = datetime.strptime(short.loc[i + window - 1].created_at, ES_DATE_FORMAT)
    if i == 0:
        short_news = news[news.date <= max_date]
