import pandas as pd
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# uniform_data = np.random.rand(10, 12)
# ax = sns.heatmap(uniform_data, linewidth=0.5)
# plt.show()


def visualize(path, count):
    results = {}
    params = []
    time = ["Jul 16 - Jul 22", "Jul 22 - Jul 27", "Jul 27 - Aug 1", "Aug 1 - Aug 6"]
    for i in range(count):
        results[i] = pd.read_csv(path.replace(".csv", "_{}.csv".format(i)))
    m = np.zeros((count, count))
    for algo in [results[0].algo.unique()[1]]:
        params = []
        for i in range(count):
            res = results[i][(results[i].algo == algo) & (results[i].similarity == 0.3)]
            m[i][i] = res.f1.max()
            p = res[res.f1 == m[i][i]].sort_values("days").iloc[0]

            params.append("\n".join([
                label + ":" + str(p[k]) for label, k in zip(
                    ["txt", "url", "htag", "days"],
                    ["weights_text", "weights_url", "weights_hashtag", "days"]
                )]))
            print(p["similarity"])
            print(p["t"])
            for j in range(count):
                if i != j:
                    other_res = results[j][(results[j].algo == algo) & (results[j].similarity == 0.3)]
                    m[j][i] = other_res[(other_res.t == p["t"])
                                        & (other_res.days == p["days"])
                                        & (other_res.weights_text == p["weights_text"])
                                        & (other_res.weights_url == p["weights_url"])
                                        & (other_res.weights_hashtag == p["weights_hashtag"])
                    ].iloc[0].f1

        ax = sns.heatmap(m, linewidth=0.5, annot=True, vmax=0.92, vmin=0.7, cmap="Blues",
                         xticklabels=params, yticklabels=time)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.savefig("/".join(path.split("/")[0:-1]) + "/figures/{}".format(algo), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path', required=False, default="/home/bmazoyer/Dev/news_twitter/results_event2018_short.csv")
    parser.add_argument('--count', required=False, default=4)
    args = vars(parser.parse_args())

    visualize(**args)

