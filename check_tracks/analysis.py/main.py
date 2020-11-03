# This is a sample Python script.

import pandas as pd
import pandasx as pdx

HOME_DIR = ".."
BY_LOC_DIR = HOME_DIR + "/encounters/by_loc"
INFECTIONS_DIR = HOME_DIR + "/infections"
INFECTED_USERS = HOME_DIR + "/infected.csv"
USERS = [200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,
         228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,
         256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,
         284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,
         312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,
         340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,364,365,366,367,368,
         369,370,371,372,373,374,375,376,377,378,379,380,400,402,403,404,405,410,411,412,413,414,415,417,420,422,424,425,
         426,428,430,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,450,451,452,455,456,458,459,462,463,465,
         466,467,468,470,471,473,474,475,478,480,481,482,483,484,485,488,489,491,492,495,496,499,501,504,510,511,512,514,
         515,517,519,522,524,525,526,527,528,530,531,532,533,534,535,539,540,541,542,544,545,547,548,553,555,559,567,568,
         572,574,579,600,603,604,610,613,614,617,622,624,625,630,634,636,637,638,639,641,642,644,645,651,652,655,658,662,
         665,666,667,668,671,674,682,683,684,685,691,692,695,701,704,711,712,715,722,725,726,728,733,734,740,742,744,767,
         768,774,800,803,804,810,817,825,852,862,865,867,868,882,884,885,892,904,915,926,928,934,940,942,944,967,968,974,
         1010,1062,1065,1068,1082,1084,1085,1104,1115,1128,1140,1142,1144,1167,1210,1282,1285,1328,1340,1342,1344,1410,
         1485,1528,1542,1728,1742,1928,2128,2328,2528,2728,2928,3128
]

GT_SIDE = 5
GT_INTERVAL = 1
GT_CONTACT_EFFICIENCY = 10

SIDES = 5, 10, 20, 50, 100
INTERVALS = 1, 5, 10, 15, 30, 60
CONTACT_EFFICIENCIES = 10, 9, 7, 5, 3, 1


def load_infected():
    df = pdx.load_data(INFECTED_USERS, header=0, dtype=[str]*20)
    print(df.head())
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
    print(df.head())
    return df
# end


def select_most_infected(infections: pd.DataFrame, day: int):
    infected = []
    for user in USERS:
        pass
    pass



def main(name="PyCharm"):
    infected = load_infected()
    gt_infections = load_infections_data()

    for side in SIDES:
        for interval in INTERVALS:
            infections = load_infections_data(side=side, interval=interval, contact_efficiency=10, index=0)

# end


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
