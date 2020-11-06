# This is a sample Python script.

import numpy as np
import pandas as pd
import pandasx as pdx
import csvx
from commons import *


def precision_list(gt_list, u_list, n_users):
    prec_list = []
    for n in n_users:
        gt = set(gt_list[0:n])
        us = set( u_list[0:n])

        tp = gt.intersection(us)
        prec = round(len(tp)/n, DIGITS)

        prec_list.append(prec)
    return prec_list
# end


def load_infected():
    df = pdx.load_data(INFECTED_USERS, header=0, dtype=[str]*20)
    # print(df.head())
    return df
# end


def load_infections_data(
        side: int = GT_SIDE,
        interval: int = GT_INTERVAL,
        contact_efficiency: int = GT_CONTACT_EFFICIENCY,
        index: int = 0):

    filename = "{dir}/{s}_{t}_{ce}/infections_{s:02}_{t:02}_{ce:02}_{i:03}.csv".format(
        dir=INFECTIONS_DIR,
        s=side, t=interval, ce=contact_efficiency, i=index
    )

    df = pdx.load_data(filename, dtype=[int, int] + [float]*398)
    # print(df.head())
    return df
# end


def sort_most_infected(infections: pd.DataFrame):
    infected = []
    for u in USERS:
        inf_prob = infections[str(u)].max()
        infected.append( (u, inf_prob) )
    pass

    # sort by infection probability
    most_infected = sorted(infected, key=lambda p: (p[1], -p[0]), reverse=True)
    return [p[0] for p in most_infected], most_infected, len([p for p in most_infected if p[1] > 0])
# end


def collect_gt(nsims):
    gt_list = []
    for index in range(nsims):
        gt_infections = load_infections_data(
            side=GT_SIDE,
            interval=GT_INTERVAL,
            contact_efficiency=GT_CONTACT_EFFICIENCY,
            index=index)
        gt_data = sort_most_infected(gt_infections)

        gt_list.append( gt_data )
    return gt_list
# end


def collect_data(sides, intervals, ceffs, nsims, gt_list):
    data = []
    for side in sides:
        for interval in intervals:
            for ce in ceffs:
                prec_mat = []
                for index in range(nsims):
                    gt_user_list, gt_most_infected, gt_n_infected = gt_list[index]

                    infections = load_infections_data(side=side, interval=interval, contact_efficiency=ce, index=index)
                    user_list, most_infected, n_infected = sort_most_infected(infections)
                    print("n infected({},{}):".format(side, interval), n_infected)

                    prec_list = precision_list(gt_user_list, user_list, [20, 25, 30, 35, 40, 45, 50])
                    prec_mat.append([n_infected] + prec_list)
                # end

                # mean & sdev for each column
                pm = np.array(prec_mat)
                means = pm.mean(axis=0)
                sdevs = pm.std(axis=0)

                # compose the list [mean1, sdev1, mean2, sdev2, ...
                mnsdv_list = []
                for i in range(len(means)):
                    mnsdv_list.append(round(means[i], DIGITS))
                    mnsdv_list.append(round(sdevs[i], DIGITS))

                data.append([side, interval, ce] + mnsdv_list)
    # end end end
    return data
# end


def main():
    # infected = load_infected()

    header = ["side", "interval", "ce",
              "mean_infected", "sdv_infected",
              "p20", "sdv20",
              "p25", "sdv25",
              "p30", "sdv30",
              "p35", "sdv35",
              "p40", "sdv40",
              "p45", "sdv45",
              "p50", "sdv50"]

    nsims = NSIMS

    gt_list = collect_gt(nsims)

    data = collect_data(SIDES, INTERVALS, CONTACT_EFFICIENCIES, nsims, gt_list)
    csvx.save_csv("infected_operator.csv", data, header)

    data = collect_data(CE_SIDES, CE_INTERVALS, CE_CONTACT_EFFICIENCIES, nsims, gt_list)
    csvx.save_csv("infected_contact_app.csv", data, header)
# end


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
