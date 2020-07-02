import pandas as pd
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

titles = ["Evaluation on all documents", "Evaluation on tweets only"]


def visualize_days(path, count):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    for enum, algo in enumerate(["louvain_macro_tfidf", "louvain_macro_tfidf tweets only"]):
        res = pd.DataFrame()
        for i in range(count):
            results = pd.read_csv(path.replace(".csv", "_{}.csv".format(i)))
            results = results[
                (results.algo == algo) & (results.weights_text == 1) & (results.similarity == 0.3) & (results.t == 0.7)]
            res = pd.concat([res, results])
        if count == 0:
            res = pd.read_csv(path)
            res = res[
                (res.algo == algo) & (res.weights_text == 1) & (res.similarity == 0.3) & (res.t == 0.7)]
        res["model"] = res.model.str.replace("VertexPartition", "")
        res["days"] = res["days"].astype(int)
        res = res.rename(columns={"days": "∆ (days)", "model": "objective function"})
        sns.pointplot(x="∆ (days)", y="f1", data=res, hue="objective function", ci="sd", ax=[ax1, ax2][enum])
        [ax1, ax2][enum].set_ylim(0.7, 0.95)
        [ax1, ax2][enum].title.set_text(titles[enum])
    if count == 0:
        plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/louvain_macro_days_all_corpus.pdf",
                    bbox_inches='tight')
    else:
        plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/louvain_macro_days.pdf",
                    bbox_inches='tight')
    plt.show()


def visualize_sim(path, count):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    for enum, algo in enumerate(["louvain_macro_tfidf", "louvain_macro_tfidf tweets only"]):
        res = pd.DataFrame()
        for i in range(count):
            results = pd.read_csv(path.replace(".csv", "_{}.csv".format(i)))
            results = results.drop_duplicates()
            results = results[
                (results.algo == algo) & (results.weights_text == 1) & (results.days == 1) & (
                            results.t == 0.7)  & (results.similarity < 0.7)]
            res = pd.concat([res, results])
        res["model"] = res.model.str.replace("VertexPartition", "")
        res = res.rename(columns={"similarity": "s", "model": "objective function"})
        sns.pointplot(x="s", y="f1", data=res, hue="objective function", ci="sd", ax=[ax1, ax2][enum])
        [ax1, ax2][enum].set_ylim(0.45, 0.95)
        [ax1, ax2][enum].title.set_text(titles[enum])
    plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/louvain_macro_sim.pdf",
                bbox_inches='tight')
    plt.show()

def visualize_modalities(path, count):
    results = {}
    time = ["Jul 16 - Jul 22", "Jul 22 - Jul 27", "Jul 27 - Aug 1", "Aug 1 - Aug 6"]
    matrix_best_params = np.zeros((count, count))
    matrix_fsd = np.zeros((count, 1))
    matrix_text_only = np.zeros((count, 1))
    tot_max = 0
    params_text_only = ['\n'.join([label + ": " + value for label, value in zip(
        ["α text", "α url", "α htag", "∆"],
        ["1", "0", "0", "1"])])]
    for i in range(count):
        results[i] = pd.read_csv(path.replace(".csv", "_{}.csv".format(i)))
        matrix_fsd[i][0] = results[i][(results[i].algo == "FSD") & (results[i].t == 0.7)].f1.max()

    for algo in ["louvain_macro_tfidf", "louvain_macro_tfidf tweets only"]:
        params = []
        for i in range(count):
            res = results[i][(results[i].algo == algo) & (results[i].similarity == 0.3)]
            matrix_text_only[i][0] = res[(results[i].weights_text == 1) & (results[i].days == 1)].f1.max()
            matrix_best_params[i][i] = res.f1.max()
            p = res[res.f1 == matrix_best_params[i][i]].sort_values("days").tail(1).iloc[0]
            param_string = "\n".join([
                label + ": " + str(p[k]) for label, k in zip(
                    ["α text", "α url", "α htag"],
                    ["weights_text", "weights_url", "weights_hashtag"]
                )])
            param_string += "\n∆: {}".format(str(int(p["days"])))
            params.append(param_string)

            for j in range(count):
                if i != j:
                    other_res = results[j][(results[j].algo == algo) & (results[j].similarity == 0.3)]
                    matrix_best_params[j][i] = other_res[(other_res.t == p["t"])
                                        & (other_res.days == p["days"])
                                        & (other_res.weights_text == p["weights_text"])
                                        & (other_res.weights_url == p["weights_url"])
                                        & (other_res.weights_hashtag == p["weights_hashtag"])
                    ].iloc[0].f1
        tot_max = max(tot_max, matrix_best_params.max())
        plt.figure(figsize=(7, 4.5))
        ax1 = plt.subplot2grid((1, 10), (0, 0), colspan=7, rowspan=1)
        ax2 = plt.subplot2grid((1, 10), (0, 7), colspan=3, rowspan=1)
        sns.heatmap(matrix_best_params, annot=True, vmax=tot_max,
                    vmin=matrix_fsd.min(), cmap="Blues",
                         xticklabels=params, yticklabels=time, square=True, fmt='.3f', ax=ax1, cbar=False)

        # plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/{}".format(algo), bbox_inches='tight')
        # plt.show()
        # plt.clf()
        sns.heatmap(matrix_text_only, annot=True, vmax=tot_max,
                    vmin=matrix_fsd.min(), cmap="Blues",
                    xticklabels=params_text_only, yticklabels=False, square=True, fmt='.3f', ax=ax2)

        plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/{}.pdf".format(algo.replace(" ", "_")),
                    bbox_inches='tight',
                    )
        plt.show()
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', required=False, default="/home/bmazoyer/Dev/news_twitter/results_event2018.csv")
    parser.add_argument('--count', required=False, default=0)
    args = vars(parser.parse_args())

    # visualize_modalities(**args)
    # visualize_sim(**args)
    visualize_days(**args)

