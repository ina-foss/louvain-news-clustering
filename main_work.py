from twembeddings.build_features_matrix import format_text, find_date_created_at, build_matrix
from twembeddings.embeddings import TfIdf
from twembeddings import ClusteringAlgoSparse
from twembeddings import general_statistics, cluster_event_match
from twembeddings.eval import cluster_acc
import logging
import sklearn.cluster
import pandas as pd
import re
import igraph as ig
import louvain
import csv
from scipy import sparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing.data import _handle_zeros_in_scale
from datetime import datetime, timedelta
import argparse

logging.basicConfig(filename='/usr/src/app/app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def cosine_similarity(x, y):
    s = normalize(x) * normalize(y).T
    return s


def find_hashtag(text):
    hashtags = []
    for word in re.split(r"[.()' \n]", text):
      if word.startswith("#"):
          hashtags.append(word.strip('&,".—/'))
    if hashtags:
      return hashtags


def zero_one_scale(serie):
    data_range = serie.max()
    scale = 1 / _handle_zeros_in_scale(data_range)
    serie *= scale


def compute_events(tweets_path, news_path, lang, binary, threshold_tweets, binary_news=False, threshold_news=0.7):

    news = pd.read_csv(news_path, sep="\t", quoting=csv.QUOTE_ALL, dtype={"id": int, "label": float,
                                                                          "created_at": str, "text": str, "url": str})
    tweets = pd.read_csv(tweets_path, sep="\t", quoting=csv.QUOTE_ALL, dtype={"id": int, "label": float,
                                                                          "created_at": str, "text": str, "url": str})
    if tweets.id.min() <= news.id.max():
        raise Exception("tweets.id.min() should be greater than news.id.max()")

    if "pred" not in news.columns:
        news_pred, news, _, X_tweets = fsd(news_path, lang, threshold=threshold_news, binary=binary_news)
        news["pred"] = news_pred
        news.id = news.id.astype(int)
    else:
        news["pred"] = news["pred"].fillna(news["id"]).astype(int)
    if "pred" not in tweets.columns:
        tweets_pred, tweets, _, X_news = fsd(tweets_path, lang, threshold=threshold_tweets, binary=binary)
        tweets["pred"] = tweets_pred
        tweets.id = tweets.id.astype(int)
        # tweets.loc[(tweets.pred == -1) | (tweets.pred == -2), "pred"] = tweets["id"]

    if tweets.pred.min() <= news.pred.max():
        tweets["pred"] = tweets["pred"] + news.pred.max() + 3
    logging.info("Total tweets: {} preds".format(tweets.pred.nunique()))
    logging.info("Total news: {} preds".format(news.pred.nunique()))
    news["type"] = "news"
    tweets["type"] = "tweets"
    return tweets, news


def louvain_macro_tfidf(tweets_path, news_path, lang, similarity, weights, binary=True,
                        threshold_tweets=0.7, model="ModularityVertexPartition", days=1):
    """
    Apply Louvain algorithm on graph of events (an event consists in all documents in the same
    fsd clusters).
    :param str tweets_path: path to the tweets dataset in format "id"    "label"   "created_at"    "text"    "url"
    "pred" is optional if fsd clustering is already done
    :param str news_path: path to the news dataset in format "id"    "label"    "created_at"    "text"    "url"
    "pred" is optional if fsd clustering is already done
    :param str lang: "fr" or "en"
    :param bool binary: if True, all non-zero term counts are set to 1 in tf calculation for tweets
    :param bool binary_news: if True, all non-zero term counts are set to 1 in tf calculation for news
    :return: y_pred, data, params
    """
    tweets, news = compute_events(tweets_path, news_path, lang, binary, threshold_tweets)
    data = pd.concat([tweets, news], ignore_index=True, sort=False)
    logging.info("save data")
    local_path = tweets_path.split("data/")[0]
    path = local_path + "data/" + (tweets_path + "_" + news_path).replace(local_path + "data/", "")
    data.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_ALL)
    args = {"dataset": path, "model": "tfidf_all_tweets", "annotation": "no", "hashtag_split": True,
            "lang": lang, "text+": False, "svd": False, "tfidf_weights": False, "save": False, "binary": False}
    data["date"] = data["created_at"].apply(find_date_created_at)
    logging.info("build matrix")
    vectorizer = TfIdf(lang=args["lang"], binary=args["binary"])
    vectorizer.load_history(args["lang"])
    data.text = data.text.apply(format_text,
                                remove_mentions=True,
                                unidecode=True,
                                lower=True,
                                hashtag_split=args["hashtag_split"]
                                )
    count_matrix = vectorizer.add_new_samples(data)
    X = vectorizer.compute_vectors(count_matrix, min_df=10, svd=args["svd"], n_components=100)
    data["hashtag"] = data.hashtag.str.split("|")
    data["url"] = data.url.str.split("|")
    # logging.info("save data")
    # data.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_ALL)
    # data = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_ALL, dtype={"date": str})
    X = X[data.pred.argsort()]
    # logging.info("save X")
    # sparse.save_npz("/usr/src/app/data/X.npz", X)
    data = data.sort_values("pred").reset_index(drop=True)
    gb = data.groupby(["pred", "type"])
    macro = gb.agg({
        'date': ['min', 'max', 'size'],
        'hashtag': lambda tdf: tdf.explode().tolist(),
        'url': lambda tdf: tdf.explode().tolist()
    })
    macro.columns = ["mindate", "maxdate", "size", "hashtag", "url"]
    macro = macro.reset_index().sort_values("pred")
    macro_tweets = macro[macro.type == "tweets"]
    macro_news = macro[macro.type == "news"]
    m = sparse.csr_matrix((
        [1 for r in range(X.shape[0])],
        ([i for i, row in macro.iterrows() for r in range(row["size"])],
         range(X.shape[0]))
    ))
    logging.info("tfidf_sum")
    tfidf_sum = m * X
    logging.info("tfidf_mean")
    for i, val in enumerate(macro["size"].tolist()):
        tfidf_sum.data[tfidf_sum.indptr[i]:tfidf_sum.indptr[i + 1]] /= val
    mean_tweets = tfidf_sum[np.array(macro.type == "tweets")]
    mean_news = tfidf_sum[np.array(macro.type == "news")]
    logging.info("load mean matrices")
    # sparse.save_npz("/usr/src/app/data/mean_tweets.npz", mean_tweets)
    # sparse.save_npz("/usr/src/app/data/mean_news.npz", mean_news)
    # mean_news = sparse.load_npz("/usr/src/app/data/mean_news.npz")
    # mean_tweets = sparse.load_npz("/usr/src/app/data/mean_tweets.npz")

    edges_text = []
    edges_hashtags = []
    edges_urls = []
    logging.info("cosine similarity")
    min_max = macro_tweets[["mindate", "maxdate"]].drop_duplicates().reset_index(drop=True)
    total = min_max.shape[0]
    for iter, row in min_max.iterrows():
        logging.info(iter/total)
        batch_min = (datetime.strptime(row["mindate"], "%Y%m%d") - timedelta(days=days)).strftime("%Y%m%d")
        batch_max = (datetime.strptime(row["maxdate"], "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")

        bool_tweets = (macro_tweets.mindate == row["mindate"]) & (macro_tweets.maxdate == row["maxdate"])
        bool_news = ((batch_min <= macro_news.maxdate) & (macro_news.maxdate <= batch_max)) | (
                (batch_min <= macro_news.maxdate) & (macro_news.maxdate <= batch_max))

        batch_tweets = macro_tweets[bool_tweets]
        batch_news = macro_news[bool_news]
        if weights["text"] != 0:
            sim = cosine_similarity(mean_tweets[np.array(bool_tweets)], mean_news[np.array(bool_news)])
            close_events = sparse.coo_matrix(sim >= similarity)
            batch = [
                (batch_tweets.iloc[i]["pred"],
                 batch_news.iloc[j]["pred"],
                 sim[i, j]
                 ) for i, j in zip(close_events.row, close_events.col)
            ]
            edges_text.extend(batch)
        if weights["hashtag"] != 0:
            hashtags_tweets = batch_tweets.explode("hashtag")
            hashtags_news = batch_news.explode("hashtag")
            # hashtags_tweets = hashtags_tweets.drop_duplicates(["pred", "hashtag"])
            # hashtags_news = hashtags_news.drop_duplicates(["pred", "hashtag"])
            hashtags_tweets = hashtags_tweets.groupby(["pred", "hashtag"]).size().reset_index(name="weight")
            hashtags_news = hashtags_news.groupby(["pred", "hashtag"]).size().reset_index(name="weight")
            batch = hashtags_tweets.merge(hashtags_news, on="hashtag", how='inner', suffixes=("_tweets", "_news"))
            # batch["weight"] = batch["weight_tweets"] + batch["weight_news"]
            batch = batch.groupby(["pred_tweets", "pred_news"])["weight_tweets"].agg(['sum', 'size']).reset_index()
            # batch = batch.groupby(["pred_tweets", "pred_news"]).size().reset_index(name="weight")
            batch = batch[batch["size"] > 3]
            edges_hashtags.extend(batch[["pred_tweets", "pred_news", "sum"]].values.tolist())
        if weights["url"] != 0:
            urls_tweets = batch_tweets.explode("url")
            urls_news = batch_news.explode("url")
            urls_tweets = urls_tweets.groupby(["pred", "url"]).size().reset_index(name="weight")
            urls_news = urls_news.groupby(["pred", "url"]).size().reset_index(name="weight")
            batch = urls_tweets.merge(urls_news, on="url", how='inner', suffixes=("_tweets", "_news"))
            # batch["weight"] = batch["weight_tweets"] + batch["weight_news"]
            batch = batch.groupby(["pred_tweets", "pred_news"])["weight_tweets"].agg(['sum', 'size']).reset_index()
            # batch = batch.groupby(["pred_tweets", "pred_news"]).size().reset_index(name="weight")
            # batch = batch[batch["size"] > 1]
            edges_urls.extend(batch[["pred_tweets", "pred_news", "sum"]].values.tolist())

    edges_hashtags = pd.DataFrame(edges_hashtags, columns=["pred_tweets", "pred_news", "weight"])
    zero_one_scale(edges_hashtags["weight"])
    edges_hashtags["weight"] *= weights["hashtag"]
    edges_urls = pd.DataFrame(edges_urls, columns=["pred_tweets", "pred_news", "weight"])
    zero_one_scale(edges_urls["weight"])
    edges_urls["weight"] *= weights["url"]
    edges_text = pd.DataFrame(edges_text, columns=["pred_tweets", "pred_news", "weight"])
    edges_text["weight"] *= weights["text"]
    edges = pd.concat([edges_text, edges_hashtags, edges_urls]).groupby(
        ["pred_tweets", "pred_news"])["weight"].sum().sort_values()
    g = ig.Graph.TupleList([(i[0], i[1], row) for i, row in edges.iteritems()],
                           weights=True)
    logging.info("build partition")
    partition = louvain.find_partition(g, getattr(louvain, model), weights="weight")
    max_pred = int(data.pred.max()) + 1
    for cluster in range(len(partition)):
        data.loc[data.pred.isin(g.vs.select(partition[cluster])["name"]), "pred"] = cluster + max_pred
    params = {"t": threshold_tweets,
              "dataset": tweets_path + " " + news_path, "algo": "louvain_macro_tfidf", "lang": lang,
              "similarity": similarity, "weights_text": weights["text"], "weights_hashtag": weights["hashtag"],
              "weights_url": weights["url"], "binary": binary, "model": model, "days": days,
              "ts": datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
    # logging.info("nb pred: {}".format(data.pred.nunique()))
    # logging.info("save to /usr/src/app/data/3_months_joint_events.csv")
    # data[["id", "pred"]].to_csv("/usr/src/app/data/3_months_joint_events.csv", quoting=csv.QUOTE_MINIMAL, index=False)
    return data.pred.tolist(), data, params


def fsd(corpus, lang, threshold, binary):
    args = {"dataset": corpus, "model": "tfidf_all_tweets", "annotation": "annotated", "hashtag_split": True,
          "lang": lang, "text+": False, "svd": False, "tfidf_weights": False, "save":False, "binary": binary}
    X, data = build_matrix(**args)
    batch_size = 8
    window = int(data.groupby("date").size().mean() // batch_size * batch_size)
    clustering = ClusteringAlgoSparse(threshold=float(threshold), window_size=window,
                                      batch_size=batch_size, intel_mkl=False)
    clustering.add_vectors(X)
    y_pred = clustering.incremental_clustering()

    params = {"t": threshold, "dataset": corpus, "algo": "FSD", "distance": "cosine", "lang": lang,
              "binary": binary, "model": model}
    return y_pred, data, params, X


def DBSCAN(corpus, lang, min_samples, eps, binary):
    args = {"dataset": corpus, "model": "tfidf_all_tweets", "annotation": "annotated", "hashtag_split": True,
            "lang": lang, "text+": False, "svd": False, "tfidf_weights": False, "save": True, "binary": binary}
    X, data = build_matrix(**args)
    logging.info("starting DBSCAN...")
    clustering = sklearn.cluster.DBSCAN(eps=eps, metric="cosine", min_samples=min_samples).fit(X)
    y_pred = clustering.labels_
    params = {"dataset": corpus, "algo": "DBSCAN", "distance": "cosine", "eps": eps, "lang": lang,
              "min_samples": min_samples, "binary": binary}
    return y_pred, data, params


def percent_linked(data):
    """
    return the share of tweets that get linked to news and the share of news that get linked to tweets
    :param data:
    :return:
    """
    data.pred = data.pred.astype(int)
    data.id = data.id.astype(int)
    tweets = data[data.type=="tweets"]
    news = data[data.type=="news"]
    pred_tweets = set(tweets.pred.unique())
    pred_news = set(news.pred.unique())
    common = pred_tweets.intersection(pred_news)
    n_linked_tweets = tweets[tweets.pred.isin(common)].shape[0]
    n_linked_news = news[news.pred.isin(common)].shape[0]
    return n_linked_tweets/tweets.shape[0], n_linked_news/news.shape[0]


def evaluate(y_pred, data, params, path, note):
    stats = general_statistics(y_pred)
    p, r, f1 = cluster_event_match(data, y_pred)
    params.update({"p": p, "r": r, "f1": f1})
    params["note"] = note
    if "news" in data.type.unique():
        linked_tweets, linked_news = percent_linked(data)
        params.update({"linked_tweets": linked_tweets, "linked_news": linked_news})
    else:
        params.pop("linked_tweets")
        params.pop("linked_news")
    stats.update(params)
    stats = pd.DataFrame(stats, index=[0])
    logging.info("\n"+ str(stats[["f1", "p", "r", "t", "similarity", "days"]]))
    try:
        results = pd.read_csv(path)
    except FileNotFoundError:
        results = pd.DataFrame()
    stats = results.append(stats, ignore_index=True)
    stats[["f1", "p", "r", "similarity", "days", "weights_text", "weights_hashtag", "weights_url", "note",
           "linked_tweets",
           "linked_news", "algo", "t", "model", "ts",
           "binary", "count", "max", "min", "mean", "50%"]].round(5).to_csv(path, index=False)
    logging.info("Saved results to {}".format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset', required=False, default="event2018")
    parser.add_argument('--path', required=False, default="")
    args = vars(parser.parse_args())

    save_results_to = "{}results_{}.csv".format(args["path"], args["dataset"])
    model = "SurpriseVertexPartition"
    note = ""
    news_dataset = "{}data/{}_news.tsv".format(args["path"], args["dataset"])
    tweets_dataset = "{}data/{}_tweets.tsv".format(args["path"], args["dataset"])

    binary = False
    for t in [0.6, 0.7]:
        for sim in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for days in [1, 2, 3, 4, 5]:
                for w in [

                    {"hashtag": 0, "text": 1, "url": 0},
                    {"hashtag": 1, "text": 0, "url": 0},
                    {"hashtag": 0, "text": 0, "url": 1},
                    {"hashtag": 0, "text": 0.9, "url": 0.1},
                    {"hashtag": 0, "text": 0.8, "url": 0.2},
                    {"hashtag": 0.1, "text": 0.8, "url": 0.1},
                    {"hashtag": 0.2, "text": 0.8, "url": 0},
                    {"hashtag": 0, "text": 0.7, "url": 0.3},
                    {"hashtag": 0.1, "text": 0.7, "url": 0.2},
                    {"hashtag": 0, "text": 0.6, "url": 0.4},
                    {"hashtag": 0.1, "text": 0.6, "url": 0.3},
                    {"hashtag": 0.2, "text": 0.6, "url": 0.2}
                ]:

                    y_pred, data, params = louvain_macro_tfidf(tweets_dataset,news_dataset,"fr",similarity=sim, weights=w,
                                                                       binary=binary,threshold_tweets=t, model=model, days=days)

                    evaluate(y_pred, data, params, save_results_to, note)
                    tweets_data = data[data.type == "tweets"].reset_index(drop=True)
                    tweets_y_pred = tweets_data.pred.tolist()
                    params["algo"] = params["algo"] + " tweets only"
                    evaluate(tweets_y_pred, tweets_data, params, save_results_to, note)
