import pandas as pd
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def visualize(path, count):
    results = {}
    time = ["Jul 16 - Jul 22", "Jul 22 - Jul 27", "Jul 27 - Aug 1", "Aug 1 - Aug 6"]
    matrix_best_params = np.zeros((count, count))
    matrix_fsd = np.zeros((count, 1))
    matrix_text_only = np.zeros((count, 1))
    tot_max = 0
    params_text_only = ['\n'.join([label + ":" + value for label, value in zip(
        ["txt", "url", "htag", "days"],
        ["1.0", "0.0", "0.0", "4.0"])])]
    for i in range(count):
        results[i] = pd.read_csv(path.replace(".csv", "_{}.csv".format(i)))
        matrix_fsd[i][0] = results[i][(results[i].algo == "FSD") & (results[i].t == 0.7)].f1.max()

    for algo in ["louvain_macro_tfidf", "louvain_macro_tfidf tweets only"]:
        params = []
        for i in range(count):
            res = results[i][(results[i].algo == algo) & (results[i].similarity == 0.3)]
            matrix_text_only[i][0] = res[(results[i].weights_text == 1) & (results[i].days == 4)].f1.max()
            matrix_best_params[i][i] = res.f1.max()
            p = res[res.f1 == matrix_best_params[i][i]].sort_values("days").tail(1).iloc[0]

            params.append("\n".join([
                label + ":" + str(p[k]) for label, k in zip(
                    ["txt", "url", "htag", "days"],
                    ["weights_text", "weights_url", "weights_hashtag", "days"]
                )]))

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
        if algo == "louvain_macro_tfidf":
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

            plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/{}".format(algo), bbox_inches='tight')
            plt.show()
            plt.clf()

    plt.figure(figsize=(8.5, 4.5))
    ax1 = plt.subplot2grid((1, 9), (0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid((1, 9), (0, 2), colspan=5, rowspan=1)
    ax3 = plt.subplot2grid((1, 9), (0, 7), colspan=2, rowspan=1)
    sns.heatmap(matrix_fsd, annot=True, vmax=tot_max, vmin=matrix_fsd.min(),
                cmap="Blues", square=True, cbar=False,
                xticklabels= False, yticklabels=time, fmt='.3f', ax=ax1)
    sns.heatmap(matrix_best_params, annot=True, vmax=tot_max, vmin=matrix_fsd.min(),
                cmap="Blues", cbar=False,
                xticklabels=params, yticklabels=False, square=True, fmt='.3f', ax=ax2)
    sns.heatmap(matrix_text_only, annot=True, vmax=tot_max, vmin=matrix_fsd.min(),
                cmap="Blues",
                xticklabels=params_text_only, yticklabels=False, square=True, fmt='.3f', ax=ax3)
    plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/{}".format(algo), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', required=False, default="/home/bmazoyer/Dev/news_twitter/results_event2018_short.csv")
    parser.add_argument('--count', required=False, default=4)
    args = vars(parser.parse_args())

    visualize(**args)

