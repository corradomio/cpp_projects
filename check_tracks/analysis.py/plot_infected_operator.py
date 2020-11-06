import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandasx as pdx
import pyplotx as pltx

from commons import *

N_AGENTS = 20, 25, 30, 35, 40, 45, 50


def load_data() -> pd.DataFrame:
    df = pdx.load_data("infected_operator.csv")
    return df
# end


def plot_data_by_side(df: pd.DataFrame,
                      sides=SIDES,
                      intervals=INTERVALS,
                      fprefix="by_side"):
    n_agents = len(N_AGENTS)
    for side in sides:
        plt.clf()
        for interval in intervals:
            prec = np.zeros(n_agents)
            sdev = np.zeros(n_agents)
            i = 0
            for n_agens in N_AGENTS:
                col_p = "p" + str(n_agens)
                sdv_p = "sdv" + str(n_agens)
                p = list(df[(df["interval"] == interval) & (df["side"] == side)][col_p])[0]
                s = list(df[(df["interval"] == interval) & (df["side"] == side)][sdv_p])[0]
                prec[i] = p
                sdev[i] = s
                i += 1
            pltx.plot(N_AGENTS, prec, sdev=sdev, label="{}min".format(interval))
        plt.ylim(0.6, 1.01)
        plt.xlabel("n top infected")
        plt.ylabel("precision")
        plt.title("Precision Infections (side: {}m, 1-sdev)".format(side))
        plt.legend()
        fname = "plots/{}_{}m.png".format(fprefix, side)
        plt.savefig(fname, dpi=300)
        # plt.show()
# end


def plot_data_by_intervals(df: pd.DataFrame,
                           sides=SIDES, intervals=INTERVALS,
                           fprefix="by_interval"):
    n_agents = len(N_AGENTS)
    for interval in intervals:
        plt.clf()
        for side in sides:
            prec = np.zeros(n_agents)
            sdev = np.zeros(n_agents)
            i = 0
            for n_agens in N_AGENTS:
                col_p = "p" + str(n_agens)
                sdv_p = "sdv" + str(n_agens)
                p = list(df[(df["interval"] == interval) & (df["side"] == side)][col_p])[0]
                s = list(df[(df["interval"] == interval) & (df["side"] == side)][sdv_p])[0]
                prec[i] = p
                sdev[i] = s
                i += 1
            pltx.plot(N_AGENTS, prec, sdev=sdev, label="{}m".format(side))
        plt.ylim(0.6, 1.01)
        plt.xlabel("n top infected")
        plt.ylabel("precision")
        plt.title("Precision Infections (interval: {}min, 1-sdev)".format(interval))
        plt.legend()
        fname = "plots/{}_{}min.png".format(fprefix, interval)
        plt.savefig(fname, dpi=300)
        # plt.show()
# end


def main():
    df = load_data()
    plot_data_by_side(df)
    plot_data_by_side(df, intervals=[5, 15, 30], fprefix="by_side_interval")
    plot_data_by_intervals(df)
    plot_data_by_intervals(df, sides=[5, 20, 100], fprefix="by_interval_side")


if __name__ == "__main__":
    main()
