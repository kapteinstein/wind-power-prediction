import numpy as np
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats
from pprint import pprint
import argparse
import datetime as dt
import gnuplotlib as gp
from itertools import cycle
import pickle
import h5py as h5
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import subprocess
import time

from db import TestResultsDatabase

import sys

style = {
    "axes.facecolor": "white",
    "axes.edgecolor": "0",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelcolor": "0",
    "figure.facecolor": "white",
}

sns.set_style("ticks", style)
sns.set_context("paper")

gnuplot_colors = [
    "#9400D3",
    "#009E73",
    "#56B4E9",
    "#E69F00",
    "#F0E442",
    "#0072B2",
    "#E51E10",
]
gnuplot_colors_uy = [
    "#9400D3",
    "#009E73",
    "#56B4E9",
    "#E69F00",
    "#0072B2",
    "#E51E10",
]  # uten gul
gnuplot_colors_uy2 = ["#9400D3", "#009E73", "#E69F00", "#0072B2", "#E51E10"]  # uten gul
gnuplot_colors_light = [
    "#DEB2F1",
    "#B2E1D5",
    "#CCE8F8",
    "#F7E2B2",
    "#FAF6C6",
    "#B2D4E7",
    "#F7BBB7",
]

palette_normal = sns.color_palette(gnuplot_colors)
palette_light = sns.color_palette(gnuplot_colors_light)
palette_2 = sns.color_palette(gnuplot_colors_uy)
palette_3 = sns.color_palette(gnuplot_colors_uy2)


sns.set_palette(palette_normal)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


mape_constants = {
    "DEU": 11635.531555175781,
    "DK1": 1250.4453,
    "DK1off": 397.86496,
    "DK1on": 929.4725,
    "DK2": 334.09128,
    "DK2off": 158.16629,
    "DK2on": 194.76888,
    "ENBW": 255.59525,
    "EONoff": 1727.7087,
    "EONon": 4061.0574,
    "FIN": 575.7175,
    "NO2": 171.41772,
    "NO3": 114.73366,
    "NO4": 91.126175,
    "NRD": 6262.307723999023,
    "RWE": 1860.2511,
    "SE1": 193.50137,
    "SE2": 681.82135,
    "SE3": 695.5517,
    "SE4": 473.62903,
    "Vattenfalloff": 231.28091,
    "Vattenfallon": 3499.6382,
}


def get_utc(timestamp):
    return dt.datetime.utcfromtimestamp(timestamp)


class NegateAction(argparse.Action):
    def __call__(self, parser, ns, values, option):
        setattr(ns, self.dest, option[2:4] != "no")


def moving_average(a, n=3):
    if len(a.shape) == 1:
        a = np.cumsum(a)
        a[n:] = a[n:] - a[:-n]
        a = a[n - 1 :] / n
    if len(a.shape) == 2:
        a = np.cumsum(a, axis=1)
        a[:, n:] = a[:, n:] - a[:, :-n]
        a = a[:, n - 1 :] / n
    return a


def calculate_aape(estimated, target, region=None):
    return np.arctan2(np.abs(estimated - target), np.abs(target))


def calculate_ape(estimated, target, region=None):
    # return np.abs(estimated - target) / mape_constants[region]
    return np.abs(estimated - target) / np.mean(target)


def calculate_ae(estimated, target, region=None):
    return np.abs(estimated - target)


def calculate_e(estimated, target, region=None):
    return estimated - target


def calculate_maape(estimated, target, region=None):
    AAPE = calculate_aape(estimated, target)
    return np.mean(AAPE, axis=1)


def calculate_mape(estimated, target, region=None):
    # if region==None:
    #     raise
    #     exit('[error] cannot calculate mape without specifying region')

    # return np.mean(np.abs(estimated - target), axis=1) / mape_constants[region]
    return np.mean(np.abs(estimated - target), axis=1) / np.mean(target)


def calculate_mae(estimated, target, region=None):
    return np.mean(np.abs(estimated - target), axis=1)


def calculate_me(estimated, target, region=None):
    return np.mean(estimated - target, axis=1)


def caclulate_statistics(db, param, verbose=0):
    print("statistics for single region")
    if verbose >= 2:
        print("filter:")
        pprint(param)
        print()

    region = param["comment"].split(".")[0]
    target = db.get(**param, target=True)
    timestamps = db.get(**param, is_timestamps=True)
    if len(target.shape) > 1:
        print("Warning. target burde ikke ha mer enn 1 rad")
        target = target[0]
    estimated = db.get(**param, target=False)
    if len(target) == 0:
        print("No targets found")
        return
    if len(estimated) == 0:
        print("No estimates found")
        return

    if len(estimated.shape) == 1:
        estimated = np.expand_dims(estimated, axis=0)

    if verbose >= 1:
        print("[Info] number of estimated series to use as basis:", estimated.shape[0])
        print()

    maape = calculate_maape(estimated, target)
    mape = calculate_mape(estimated, target, region=region)
    mae = calculate_mae(estimated, target)
    me = calculate_me(estimated, target)

    if args.figures:
        navg = 627  # one month moving avg
        fig = plt.figure(figsize=(16, 12))
        plt.plot(target[navg:-navg])
        plt.savefig("out/target-{}.pdf".format(region))
        plt.close(fig)

        fig = plt.figure(figsize=(16, 12))
        for series in estimated:
            plt.plot(series[1000:1500], alpha=0.5, lw=0.1, c="k")
        plt.plot(target[1000:1500])
        plt.savefig("out/target_estimations-{}.pdf".format(region))
        plt.close(fig)

        fig = plt.figure(figsize=(16, 12))
        aape = calculate_aape(estimated, target)
        moving_average_aape = moving_average(aape, n=672)
        for i, series in enumerate(moving_average_aape):
            if i == np.argsort(maape)[0]:
                plt.plot(series, lw=2, label="0")
            elif i == np.argsort(maape)[1]:
                plt.plot(series, lw=2, label="1")
            elif i == np.argsort(maape)[2]:
                plt.plot(series, lw=2, label="2")
            else:
                plt.plot(series, c="k", alpha=0.2)
        plt.legend()
        plt.savefig("out/moving_average_aape-{}.pdf".format(region))
        plt.close(fig)

        aape = calculate_aape(estimated, target)
        ape = calculate_ape(estimated, target, region=region)
        ae = calculate_ae(estimated, target)
        e = calculate_e(estimated, target)
        idx_sorted = np.argsort(target)
        # gp.plot(target[idx_sorted], aape[0][idx_sorted], _with="points")
        # gp.plot(target[idx_sorted], ape[0][idx_sorted], _with="points")
        # gp.plot(target[idx_sorted], ae[0][idx_sorted], _with="points")
        # gp.plot(target[idx_sorted], e[0][idx_sorted], _with="points")

        # sns.boxplot(x=['aape', 'ape', 'ae', 'e'], y=[aape, ape, ae, e])
        # plt.show()
        # exit()
        # gp.plot((timestamps[3500:3700], target[3500:3700]), (timestamps[3500:3700], estimated[0][3500:3700]), _with="line")

        fig = plt.figure()
        sns.distplot(target, label="target", norm_hist=True)
        sns.distplot(estimated[0], label="estimated", norm_hist=True)
        plt.title("distribution of production - {}".format(param["comment"]))
        plt.ylabel("frequency")
        plt.xlabel("production")
        plt.legend()
        plt.savefig("out/distribution-{}.pdf".format(region))
        plt.close(fig)

    if verbose >= 1:
        print(
            "[Info] timestamps: {} to {}".format(
                get_utc(timestamps[0]), get_utc(timestamps[-1])
            )
        )

    if estimated.shape[0] == 1:
        print("[Info] standard deviation of the mean does not apply to only one series")
        print("MAAPE mean: {:.3}".format(np.mean(maape)))
        print("MAPE  mean: {:.3}".format(np.mean(mape)))
        print("MAE   mean: {:.3}".format(np.mean(mae)))
        print("ME    mean: {:.3}".format(np.mean(me)))
        caclulate_statistics_validation(db, param, verbose)
        return
    print("MAAPE std*2: {:>8.3f}".format(np.std(maape) * 2))
    print("MAAPE standard deviation: {:.3}".format(np.std(maape)))
    print("MAPE  standard deviation: {:.3}".format(np.std(mape)))
    print("MAE   standard deviation: {:.3}".format(np.std(mae)))
    print("ME    standard deviation: {:.3}".format(np.std(me)))
    print()
    print(
        "MAAPE 0.95 confidence interval (SEM): ({:.3}, {:.3}),\tmean={:.3}".format(
            np.mean(maape) - 2 * scipy.stats.sem(maape),
            np.mean(maape) + 2 * scipy.stats.sem(maape),
            np.mean(maape),
        )
    )
    print(
        "MAPE  0.95 confidence interval (SEM): ({:.3}, {:.3}),\tmean={:.3}".format(
            np.mean(mape) - 2 * scipy.stats.sem(mape),
            np.mean(mape) + 2 * scipy.stats.sem(mape),
            np.mean(mape),
        )
    )
    print(
        "MAE   0.95 confidence interval (SEM): ({:.3}, {:.3}),\tmean={:.3}".format(
            np.mean(mae) - 2 * scipy.stats.sem(mae),
            np.mean(mae) + 2 * scipy.stats.sem(mae),
            np.mean(mae),
        )
    )
    print(
        "ME    0.95 confidence interval (SEM): ({:.3}, {:.3}),\tmean={:.3}".format(
            np.mean(me) - 2 * scipy.stats.sem(me),
            np.mean(me) + 2 * scipy.stats.sem(me),
            np.mean(me),
        )
    )

    caclulate_statistics_validation(db, param, verbose)


def caclulate_statistics_validation(db, param, verbose=0):
    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK2",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]
    regions = regions_ger + regions_nrd

    _metric = args.rank_metric
    print("validation for single region")
    if verbose >= 2:
        print("filter:")
        pprint(param)
        print()

    is_hybrid = param["hybrid"]
    region = param["comment"].split(".")[0]
    version = param["comment"].split(".")[-1]
    other_version = "4" if version == "5" else "5"

    # if is_hybrid:
    #    param['hybrid'] = False
    valid_target = db.get(**param, is_validation=True, target=True)
    valid_timestamps = db.get(**param, is_validation=True, is_timestamps=True)
    if len(valid_target.shape) > 1:
        print("Warning. target burde ikke ha mer enn 1 rad")
        valid_target = valid_target[0]
    valid_estimated = db.get(**param, is_validation=True, target=False)
    if len(valid_target) == 0:
        print("No targets found")
        return
    if len(valid_estimated) == 0:
        print("No estimates found")
        return

    if len(valid_estimated.shape) == 1:
        valid_estimated = np.expand_dims(valid_estimated, axis=0)

    # if is_hybrid:
    #    param['hybrid'] = True
    target = db.get(**param, target=True)
    timestamps = db.get(**param, is_timestamps=True)
    if len(target.shape) > 1:
        print("Warning. target burde ikke ha mer enn 1 rad")
        target = target[0]
    estimated = db.get(**param, target=False)
    if len(target) == 0:
        print("No targets found")
        return
    if len(estimated) == 0:
        print("No estimates found")
        return

    if len(estimated.shape) == 1:
        estimated = np.expand_dims(estimated, axis=0)

    param["hybrid"] = True
    hybrid_valid_target = db.get(**param, is_validation=True, target=True)
    hybrid_valid_timestamps = db.get(**param, is_validation=True, is_timestamps=True)

    if len(hybrid_valid_timestamps.shape) > 1:
        print(hybrid_valid_timestamps)
        print("Warning. timestamps burde ikke ha mer enn 1 rad")
        hybrid_valid_timestamps = hybrid_valid_timestamps[0]
    if len(hybrid_valid_target.shape) > 1:
        print(hybrid_valid_target)
        print("Warning. target burde ikke ha mer enn 1 rad")
        hybrid_valid_target = hybrid_valid_target[0]
    hybrid_valid_timestamps = hybrid_valid_timestamps[2:-2]
    hybrid_valid_target = hybrid_valid_target[2:-2]
    hybrid_valid_estimated = db.get(**param, is_validation=True, target=False)
    if len(hybrid_valid_target) == 0:
        print("No targets found")
        return
    if len(hybrid_valid_estimated) == 0:
        print("No estimates found")
        return

    if len(hybrid_valid_estimated.shape) == 1:
        hybrid_valid_estimated = np.expand_dims(hybrid_valid_estimated[2:-2], axis=0)
    else:
        hybrid_valid_estimated = hybrid_valid_estimated[:, 2:-2]

    hybrid_target = db.get(**param, target=True)
    hybrid_timestamps = db.get(**param, is_timestamps=True)

    if len(hybrid_timestamps.shape) > 1:
        print("Warning. timestamps burde ikke ha mer enn 1 rad")
        hybrid_timestamps = hybrid_timestamps[0]

    if len(hybrid_target.shape) > 1:
        print("Warning. target burde ikke ha mer enn 1 rad")
        hybrid_target = hybrid_target[0]
    hybrid_timestamps = hybrid_timestamps[3::]
    hybrid_target = hybrid_target[3::]
    hybrid_estimated = db.get(**param, target=False)

    if len(hybrid_target) == 0:
        print("No targets found")
        return
    if len(hybrid_estimated) == 0:
        print("No estimates found")
        return

    if len(hybrid_estimated.shape) == 1:
        hybrid_estimated = np.expand_dims(hybrid_estimated[3::], axis=0)
    else:
        hybrid_estimated = hybrid_estimated[:, 3::]

    if region in regions_ger:
        estimated *= 1  # 1.05
    if region in regions_nrd:
        estimated *= 1  # 1.03

    valid_estimated_baseline = [None] * 4
    valid_target_baseline = [None] * 4
    valid_timestamps_baseline = [None] * 4

    estimated_baseline = [None] * 4
    target_baseline = [None] * 4
    timestamps_baseline = [None] * 4

    baseline_comments = [".4.6", ".4.8", ".4.a", ".4.c"]

    for i, comment in enumerate(baseline_comments):
        param["comment"] = region + comment
        param["hybrid"] = False
        if is_hybrid:
            param["hybrid"] = False
        valid_target_baseline[i] = db.get(
            **param, is_validation=True, target=True, is_lightgbm=True
        )
        valid_timestamps_baseline[i] = db.get(
            **param, is_validation=True, is_timestamps=True, is_lightgbm=True
        )
        valid_target_baseline[i] = valid_target_baseline[i][2:-2]
        valid_timestamps_baseline[i] = valid_timestamps_baseline[i][2:-2]
        if len(valid_target_baseline[i].shape) > 1:
            print("Warning. target burde ikke ha mer enn 1 rad")
            valid_target_baseline[i] = valid_target_baseline[i][0]
        valid_estimated_baseline[i] = db.get(
            **param, is_validation=True, target=False, is_lightgbm=True
        )

        if len(valid_target_baseline[i]) == 0:
            print("No targets found")
            return
        if len(valid_estimated_baseline[i]) == 0:
            print("No estimates found")
            return

        if len(valid_estimated_baseline[i].shape) == 1:
            valid_estimated_baseline[i] = np.expand_dims(
                valid_estimated_baseline[i][2:-2], axis=0
            )
        else:
            valid_estimated_baseline[i] = valid_estimated_baseline[i][2:-2]

        target_baseline[i] = db.get(
            **param, is_validation=False, target=True, is_lightgbm=True
        )
        timestamps_baseline[i] = db.get(
            **param, is_validation=False, is_timestamps=True, is_lightgbm=True
        )
        target_baseline[i] = target_baseline[i][3::]
        timestamps_baseline[i] = timestamps_baseline[i][3::]
        if len(target_baseline[i].shape) > 1:
            print("Warning. target burde ikke ha mer enn 1 rad")
            target_baseline[i] = target_baseline[i][0]
        estimated_baseline[i] = db.get(
            **param, is_validation=False, target=False, is_lightgbm=True
        )
        if len(target_baseline[i]) == 0:
            print("No targets found")
            return
        if len(estimated_baseline[i]) == 0:
            print("No estimates found")
            return

        if len(estimated_baseline[i].shape) == 1:
            estimated_baseline[i] = np.expand_dims(estimated_baseline[i][3::], axis=0)
        else:
            estimated_baseline[i] = estimated_baseline[i][3::]

        if region in regions_ger:
            estimated_baseline[i] *= 1  # 1.05
        if region in regions_nrd:
            estimated_baseline[i] *= 1  # 1.03

    valid_timestamps = hybrid_valid_timestamps

    if verbose >= 1:
        print(
            "[Info] valid timestamps: {} to {}".format(
                get_utc(valid_timestamps[0]), get_utc(valid_timestamps[-1])
            )
        )
        print(
            "[Info] test timestamps.: {} to {}".format(
                get_utc(timestamps[0]), get_utc(timestamps[-1])
            )
        )
        print("[Info] number of estimated series to use as basis:", estimated.shape[0])
        print(
            "[Info] number of validation series to use as basis:",
            valid_estimated.shape[0],
        )
        print()

    short = {"MAAPE": "AAPE", "MAPE": "APE", "MAE": "Absolute error", "ME": "Error"}

    valid_maape = calculate_maape(valid_estimated, valid_target)
    valid_mape = calculate_mape(valid_estimated, valid_target, region=region)
    valid_mae = calculate_mae(valid_estimated, valid_target)
    valid_me = calculate_me(valid_estimated, valid_target)

    maape = calculate_maape(estimated, target)
    mape = calculate_mape(estimated, target, region=region)
    mae = calculate_mae(estimated, target)
    me = calculate_me(estimated, target)

    hybrid_valid_maape = calculate_maape(hybrid_valid_estimated, hybrid_valid_target)
    hybrid_valid_mape = calculate_mape(
        hybrid_valid_estimated, hybrid_valid_target, region=region
    )
    hybrid_valid_mae = calculate_mae(hybrid_valid_estimated, hybrid_valid_target)
    hybrid_valid_me = calculate_me(hybrid_valid_estimated, hybrid_valid_target)

    hybrid_maape = calculate_maape(hybrid_estimated, hybrid_target)
    hybrid_mape = calculate_mape(hybrid_estimated, hybrid_target, region=region)
    hybrid_mae = calculate_mae(hybrid_estimated, hybrid_target)
    hybrid_me = calculate_me(hybrid_estimated, hybrid_target)

    lightgbm_maape = calculate_mape(estimated_baseline[0], target_baseline[0])
    rf_maape = calculate_mape(estimated_baseline[1], target_baseline[1])
    knn_maape = calculate_mape(estimated_baseline[2], target_baseline[2])
    adaboost_maape = calculate_mape(estimated_baseline[3], target_baseline[3])

    print("[hybrid] MAAPE std*2: {:>8.3f}".format(np.std(hybrid_maape) * 2))
    print("[hybrid] MAAPE standard deviation: {:.3}".format(np.std(hybrid_maape)))
    print("[hybrid] MAPE  standard deviation: {:.3}".format(np.std(hybrid_mape)))
    print("[hybrid] MAE   standard deviation: {:.3}".format(np.std(hybrid_mae)))
    print("[hybrid] ME    standard deviation: {:.3}".format(np.std(hybrid_me)))
    print("[hybrid] MAAPE avg = {:>8.3f}".format(np.mean(hybrid_maape)))
    print("[lightgbm] MAAPE avg = {:}".format(lightgbm_maape))
    print("[rf]       MAAPE avg = {:}".format(rf_maape))
    print("[knn]      MAAPE avg = {:}".format(knn_maape))
    print("[AdaBoost] MAAPE avg = {:}".format(adaboost_maape))
    print()

    if args.latex:
        print("\\num{{{:.3f} \\pm {:.3f}}}".format(np.mean(maape), 2 * np.std(maape)))
        print(
            "\\num{{{:.3f} \\pm {:.3f}}}".format(
                np.mean(hybrid_maape), 2 * np.std(hybrid_maape)
            )
        )

    t, p = scipy.stats.ttest_1samp(maape, lightgbm_maape)
    print(
        "{:>16} CNN mean...: {:>8.3}, baseline mean: {:>8.3f}, mean diff: {:8.3f}, p={:8.4f}".format(
            region,
            np.mean(maape),
            lightgbm_maape[0],
            np.mean(maape - lightgbm_maape[0]),
            p[0],
        )
    )

    t, p = scipy.stats.ttest_rel(hybrid_maape, maape)
    print(
        "{:>16} hybrid mean: {:>8.3}, CNN mean: {:>8.3f}, mean diff: {:8.3f}, p={:8.4}".format(
            region,
            np.mean(hybrid_maape),
            np.mean(maape),
            np.mean(hybrid_maape - maape),
            p,
        )
    )
    t, p = scipy.stats.ttest_1samp(hybrid_maape, lightgbm_maape)
    # print(region, np.mean(maape), lightgbm_maape[0], np.mean(maape-lightgbm_maape[0]), p)
    print(
        "{:>16} hybrid mean: {:>8.3}, baseline mean: {:>8.3f}, mean diff: {:8.3f}, p={:8.4f}".format(
            region,
            np.mean(hybrid_maape),
            lightgbm_maape[0],
            np.mean(hybrid_maape - lightgbm_maape[0]),
            p[0],
        )
    )

    if estimated.shape[0] == 1:
        print("[Info] standard deviation of the mean does not apply to only one series")
        print("MAAPE mean: {:.3}".format(np.mean(valid_maape)))
        print("MAPE  mean: {:.3}".format(np.mean(valid_mape)))
        print("MAE   mean: {:.3}".format(np.mean(valid_mae)))
        print("ME    mean: {:.3}".format(np.mean(valid_me)))

        print("[Info] standard deviation of the mean does not apply to only one series")
        print("MAAPE mean: {:.3}".format(np.mean(hybrid_valid_maape)))
        print("MAPE  mean: {:.3}".format(np.mean(hybrid_valid_mape)))
        print("MAE   mean: {:.3}".format(np.mean(hybrid_valid_mae)))
        print("ME    mean: {:.3}".format(np.mean(hybrid_valid_me)))
        return
    # score system. The runs that does is best on all metrics overall during validation is chosen
    scores = np.zeros(len(valid_maape))
    hybrid_scores = np.zeros(len(hybrid_valid_maape))
    assert len(valid_maape) == len(hybrid_valid_maape)
    penalty = np.arange(len(valid_maape))

    metric_partial = None
    metric_global = None
    if _metric == "maape":
        scores[np.argsort(valid_maape)] += penalty
        hybrid_scores[np.argsort(hybrid_valid_maape)] += penalty
        metric_partial = calculate_aape
        metric_global = calculate_maape
    elif _metric == "mape":
        scores[np.argsort(valid_mape)] += penalty
        hybrid_scores[np.argsort(hybrid_valid_mape)] += penalty
        metric_partial = calculate_ape
        metric_global = calculate_mape
    elif _metric == "mae":
        scores[np.argsort(valid_mae)] += penalty
        hybrid_scores[np.argsort(hybrid_valid_mae)] += penalty
        metric_partial = calculate_ae
        metric_global = calculate_mae
    elif _metric == "me":
        scores[np.argsort(np.abs(valid_me))] += penalty
        hybrid_scores[np.argsort(hybrid_valid_me)] += penalty
        metric_partial = calculate_e
        metric_global = calculate_me
    else:
        exit("metric must be in [maape, mape, mae, me]")

    rank = np.argsort(scores)
    hybrid_rank = np.argsort(hybrid_scores)

    if verbose >= 1:
        print("[Info] rank.......:", rank)
        print("[Info] hybrid rank:", hybrid_rank)

    if verbose >= 2:
        print("[Debug] estimated.shape:", estimated.shape)
        print("[Debug] valid_estimated.shape:", valid_estimated.shape)
    estimated_total = np.append(valid_estimated, estimated, axis=1)
    target_total = np.append(valid_target, target)
    timestamp_total = np.append(valid_timestamps, timestamps)
    # timestamp_total = np.append(valid_timestamps_baseline[0], timestamps_baseline[0])

    f = h5.File("dataset/{}.h5".format(region), "r")
    load_factor = f["target/ratio"][-len(target_total) : :]
    f.close()

    load_factor[np.isnan(load_factor)] = np.mean(load_factor[~np.isnan(load_factor)])

    hybrid_estimated_total = np.append(hybrid_valid_estimated, hybrid_estimated, axis=1)
    hybrid_target_total = np.append(hybrid_valid_target, hybrid_target)

    timestamps_total_datetime = [
        datetime.datetime.utcfromtimestamp(t) for t in timestamp_total
    ]
    timestamps_total_labels = [
        t.strftime("%Y-%b-%d") for t in timestamps_total_datetime
    ]

    timestamps_test_datetime = [
        datetime.datetime.utcfromtimestamp(t) for t in timestamps
    ]
    timestamps_test_labels = [t.strftime("%Y-%b-%d") for t in timestamps_test_datetime]

    if verbose >= 2:
        print(f"[Debug] {timestamps.shape=}")
        print(f"[Debug] {timestamps_baseline[0].shape=}")
        print(f"[Debug] {hybrid_timestamps.shape=}")

    if is_hybrid:
        hybrid_adjustment = 0
    else:
        hybrid_adjustment = 3
    if verbose >= 2:
        print(f"[Debug] {timestamps.shape=}")
        print(f"[Debug] {timestamps_baseline[0].shape=}")

        print(f"[Debug] {estimated.shape=}")
        print(f"[Debug] {hybrid_estimated.shape=}")

        print(f"[Debug] {len(valid_timestamps)=}")

        print(f"[Debug] {len(hybrid_valid_timestamps)=}")
        print(f"[Debug] {len(valid_timestamps_baseline[0])=}")

        print(f"[Debug] {valid_timestamps_baseline[0][0]=}")
        print(f"[Debug] {hybrid_valid_timestamps[0]=}")
        print(f"[Debug] {valid_timestamps[0]=}")
        print(f"[Debug] {timestamps[0]=}")
        print(f"[Debug] {len(timestamp_total)=}")
        print(f"[Debug] {len(estimated_total[0])=}")
        print(f"[Debug] {len(hybrid_target_total)=}")

        print(f"[Debug] {valid_estimated.shape=}")
        print(f"[Debug] {hybrid_valid_estimated.shape=}")
    assert timestamps[0] == timestamps_baseline[0][0]
    assert hybrid_timestamps[0] == timestamps_baseline[0][0]
    assert timestamps[-1] == timestamps_baseline[0][-1]
    assert hybrid_timestamps[-1] == timestamps_baseline[0][-1]

    assert len(target) == len(hybrid_target)
    assert estimated.shape == hybrid_estimated.shape

    # print(valid_estimated_baseline.shape)
    # print(estimated_baseline.shape)

    estimated_total_baseline = [None] * 4
    target_total_baseline = [None] * 4
    for i in range(4):
        estimated_total_baseline[i] = np.append(
            valid_estimated_baseline[i], estimated_baseline[i]
        )
        target_total_baseline[i] = np.append(
            valid_target_baseline[i], target_baseline[i]
        )

    # fig = plt.figure(figsize=(16, 12))
    # aape = calculate_aape(estimated, target)
    # moving_average_aape = moving_average(aape, n=672)

    # for i, series in enumerate(moving_average_aape):
    #     if i == rank[0]:
    #         plt.plot(series, lw=2, label="valid 0")
    #     if i == rank[1]:
    #         plt.plot(series, lw=2, label="valid 1")
    #     if i == rank[2]:
    #         plt.plot(series, lw=2, label="valid 2")
    #     else:
    #         plt.plot(series, c="k", alpha=0.2)

    # plt.savefig("out/moving_average_aape_validation.pdf")
    # plt.close(fig)

    if verbose >= 2:
        print(np.argsort(scores)[0:10])

    if verbose >= 1:
        print("maape for rank 0:", maape[rank[0]])
        print(" mape for rank 0:", mape[rank[0]])
        print("  mae for rank 0:", mae[rank[0]])
        print()
        print("avg maape for top 5:", np.mean(maape[rank[0:5]]))
        print(" avg mape for top 5:", np.mean(mape[rank[0:5]]))
        print("  avg mae for top 5:", np.mean(mae[rank[0:5]]))

        print("hybrid maape for rank 0:", hybrid_maape[hybrid_rank[0]])
        print("hybrid  mape for rank 0:", hybrid_mape[hybrid_rank[0]])
        print("hybrid   mae for rank 0:", hybrid_mae[hybrid_rank[0]])
        print()
        print("hybrid avg maape for top 5:", np.mean(hybrid_maape[hybrid_rank[0:5]]))
        print("hybrid  avg mape for top 5:", np.mean(hybrid_mape[hybrid_rank[0:5]]))
        print("hybrid   avg mae for top 5:", np.mean(hybrid_mae[hybrid_rank[0:5]]))

    if is_hybrid:
        h = "hybrid"
    else:
        h = ""

    if args.figures:
        try:
            with open("out/valid_maape{}{}.pk".format(version, h), "rb") as f:
                valid_maape_store = pickle.load(f)
            with open("out/test_maape{}{}.pk".format(version, h), "rb") as f:
                test_maape_store = pickle.load(f)
        except:
            valid_maape_store = {}
            test_maape_store = {}
        fig = plt.figure()
        sns.regplot(valid_maape, maape)
        plt.title("Testing vs validation MAAPE - {}".format(region))
        plt.xlabel("Validation")
        plt.ylabel("Testing")
        plt.savefig(
            "out/validation_vs_testing_MAAPE-{}{}{}.pdf".format(region, version, h)
        )
        # plt.show()
        plt.close(fig)

        valid_maape_store[region] = valid_maape
        test_maape_store[region] = maape

        with open("out/valid_maape{}{}.pk".format(version, h), "wb") as f:
            pickle.dump(valid_maape_store, f)
            time.sleep(1)
        with open("out/test_maape{}{}.pk".format(version, h), "wb") as f:
            pickle.dump(test_maape_store, f)
            time.sleep(1)

        # fig = plt.figure()
        # sns.regplot(valid_mape, mape)
        # plt.title("MAPE")
        # plt.xlabel("validation")
        # plt.ylabel("testing")
        # plt.savefig('out/validation_vs_testing_MAPE-{}.pdf'.format(region))
        # plt.close(fig)

    estimated_top5 = np.zeros_like(target_total)
    estimated_top5_validation = np.zeros_like(valid_target)
    estimated_top5_test = np.zeros_like(target)
    for i in rank[0:5]:
        estimated_top5 += estimated_total[i] / 5
        estimated_top5_validation += valid_estimated[i] / 5
        estimated_top5_test += estimated[i] / 5

    hybrid_estimated_top5 = np.zeros_like(hybrid_target_total)
    hybrid_estimated_top5_validation = np.zeros_like(hybrid_valid_target)
    hybrid_estimated_top5_test = np.zeros_like(hybrid_target)
    for i in hybrid_rank[0:5]:
        hybrid_estimated_top5 += hybrid_estimated_total[i] / 5
        hybrid_estimated_top5_validation += hybrid_valid_estimated[i] / 5
        hybrid_estimated_top5_test += hybrid_estimated[i] / 5

    hybrid_from_cnn_estimated_top5 = np.zeros_like(hybrid_target_total)
    hybrid_from_cnn_estimated_top5_validation = np.zeros_like(hybrid_valid_target)
    hybrid_from_cnn_estimated_top5_test = np.zeros_like(hybrid_target)
    for i in rank[0:5]:
        hybrid_from_cnn_estimated_top5 += hybrid_estimated_total[i] / 5
        hybrid_from_cnn_estimated_top5_validation += hybrid_valid_estimated[i] / 5
        hybrid_from_cnn_estimated_top5_test += hybrid_estimated[i] / 5

    estimated_top5_test = np.expand_dims(estimated_top5_test, 0)
    maape = calculate_maape(estimated_top5_test, target)
    mape = calculate_mape(estimated_top5_test, target, region=region)
    mae = calculate_mae(estimated_top5_test, target)
    me = calculate_me(estimated_top5_test, target)
    estimated_top5_test = estimated_top5_test[0]

    t, p = scipy.stats.ttest_1samp(hybrid_maape, maape[0])
    print(
        "{:>16} hybrid mean: {:>8.3}, CNN top5 mean: {:>8.3f}, mean diff: {:8.3f}, p={:8.4}".format(
            region, np.mean(hybrid_maape), maape[0], np.mean(hybrid_maape - maape[0]), p
        )
    )

    hybrid_estimated_top5_test = np.expand_dims(hybrid_estimated_top5_test, 0)
    hybrid_maape = calculate_maape(hybrid_estimated_top5_test, hybrid_target)
    hybrid_mape = calculate_mape(
        hybrid_estimated_top5_test, hybrid_target, region=region
    )
    hybrid_mae = calculate_mae(hybrid_estimated_top5_test, hybrid_target)
    hybrid_me = calculate_me(hybrid_estimated_top5_test, hybrid_target)
    hybrid_estimated_top5_test = hybrid_estimated_top5_test[0]

    hybrid_from_cnn_estimated_top5_test = np.expand_dims(
        hybrid_from_cnn_estimated_top5_test, 0
    )
    hybrid_from_cnn_maape = calculate_maape(
        hybrid_from_cnn_estimated_top5_test, hybrid_target
    )
    hybrid_from_cnn_mape = calculate_mape(
        hybrid_from_cnn_estimated_top5_test, hybrid_target, region=region
    )
    hybrid_from_cnn_mae = calculate_mae(
        hybrid_from_cnn_estimated_top5_test, hybrid_target
    )
    hybrid_from_cnn_me = calculate_me(
        hybrid_from_cnn_estimated_top5_test, hybrid_target
    )
    hybrid_from_cnn_estimated_top5_test = hybrid_from_cnn_estimated_top5_test[0]

    print(hybrid_from_cnn_maape)
    # print("hybrid top5 better than cnn top 5:", hybrid_from_cnn_maape[0] < maape[0])
    print(
        "hybrid top 5 better than baseline 1:",
        hybrid_from_cnn_mape[0] < lightgbm_maape[0],
    )
    print("hybrid top 5 better than baseline 2:", hybrid_from_cnn_mape[0] < rf_maape[0])
    print(
        "hybrid top 5 better than baseline 3:", hybrid_from_cnn_mape[0] < knn_maape[0]
    )
    print(
        "hybrid top 5 better than baseline 4:",
        hybrid_from_cnn_mape[0] < adaboost_maape[0],
    )

    # try:
    #     with open('out/asdf.pk', 'rb') as f:
    #         scores = pickle.load(f)
    # except:
    #     scores = np.zeros((2,2))

    # with open('out/asdf.pk', 'wb') as f:
    #     scores[0,0] += hybrid_from_cnn_maape[0] < maape[0]
    #     scores[0,1] += hybrid_from_cnn_maape[0] > maape[0]
    #     scores[1,0] += hybrid_from_cnn_maape[0] < lightgbm_maape[0]
    #     scores[1,1] += hybrid_from_cnn_maape[0] > lightgbm_maape[0]
    #     pickle.dump(scores, f)
    #     time.sleep(1)

    # print(scores)

    print(estimated_top5.shape)
    print(hybrid_estimated_top5.shape)

    np.savetxt(
        "out/top5-test-{}{}.csv".format(region, version),
        np.stack((timestamps, estimated_top5_test, target)).T,
    )
    np.savetxt(
        "out/test-{}{}.csv".format(region, version),
        np.vstack([timestamps, estimated, target]).T,
    )
    np.savetxt(
        "out/top5-test-{}{}hybrid.csv".format(region, version),
        np.stack((timestamps, hybrid_estimated_top5_test, target)).T,
    )
    np.savetxt(
        "out/test-{}{}hybrd.csv".format(region, version),
        np.vstack([timestamps, hybrid_estimated, target]).T,
    )

    print("aggrigated top 5 validation models on test set:")
    print("maape:", maape)
    print("mape.:", mape)
    print("mae..:", mae)
    print("me...:", me)

    print("aggrigated top 5 hybrid models on test set:")
    print("maape:", hybrid_maape)
    print("mape.:", hybrid_mape)
    print("mae..:", hybrid_mae)
    print("me...:", hybrid_me)

    print(
        "aggrigated top 5 hybrid models based on the cnn validation models on test set:"
    )
    print("maape:", hybrid_from_cnn_maape)
    print("mape.:", hybrid_from_cnn_mape)
    print("mae..:", hybrid_from_cnn_mae)
    print("me...:", hybrid_from_cnn_me)

    if args.figures:
        aape_top5_total = metric_partial(estimated_top5, target_total, region=region)
        aape_top5_validation = metric_partial(
            estimated_top5_validation, valid_target, region=region
        )
        aape_top5_test = metric_partial(estimated_top5_test, target, region=region)

        hybrid_aape_top5_total = metric_partial(
            hybrid_estimated_top5, hybrid_target_total, region=region
        )
        hybrid_aape_top5_validation = metric_partial(
            hybrid_estimated_top5_validation, hybrid_valid_target, region=region
        )
        hybrid_aape_top5_test = metric_partial(
            hybrid_estimated_top5_test, hybrid_target, region=region
        )

        hybrid_from_cnn_aape_top5_total = metric_partial(
            hybrid_from_cnn_estimated_top5, hybrid_target_total, region=region
        )
        hybrid_from_cnn_aape_top5_validation = metric_partial(
            hybrid_from_cnn_estimated_top5_validation,
            hybrid_valid_target,
            region=region,
        )
        hybrid_from_cnn_aape_top5_test = metric_partial(
            hybrid_from_cnn_estimated_top5_test, hybrid_target, region=region
        )

        aape_baseline_total = [None] * 4
        aape_baseline_validation = [None] * 4
        aape_baseline_test = [None] * 4

        for i in range(4):
            aape_baseline_total[i] = metric_partial(
                estimated_total_baseline[i], target_total_baseline[i], region=region
            )
            aape_baseline_validation[i] = metric_partial(
                valid_estimated_baseline[i], valid_target_baseline[i], region=region
            )
            aape_baseline_test[i] = metric_partial(
                estimated_baseline[i], target_baseline[i], region=region
            )

        aape_total = metric_partial(estimated_total, target_total, region=region)
        aape_validation = metric_partial(valid_estimated, valid_target, region=region)
        aape_test = metric_partial(estimated, target, region=region)

        hybrid_aape_total = metric_partial(
            hybrid_estimated_total, hybrid_target_total, region=region
        )
        hybrid_aape_validation = metric_partial(
            hybrid_valid_estimated, hybrid_valid_target, region=region
        )
        hybrid_aape_test = metric_partial(
            hybrid_estimated, hybrid_target, region=region
        )

        moving_average_top5_total = moving_average(aape_top5_total, n=672)
        moving_average_aape_top5_validation = moving_average(
            aape_top5_validation, n=672
        )
        moving_average_aape_top5_test = moving_average(aape_top5_test, n=672)

        hybrid_moving_average_top5_total = moving_average(hybrid_aape_top5_total, n=672)
        hybrid_moving_average_aape_top5_validation = moving_average(
            hybrid_aape_top5_validation, n=672
        )
        hybrid_moving_average_aape_top5_test = moving_average(
            hybrid_aape_top5_test, n=672
        )

        hybrid_from_cnn_moving_average_top5_total = moving_average(
            hybrid_from_cnn_aape_top5_total, n=672
        )
        hybrid_from_cnn_moving_average_aape_top5_validation = moving_average(
            hybrid_from_cnn_aape_top5_validation, n=672
        )
        hybrid_from_cnn_moving_average_aape_top5_test = moving_average(
            hybrid_from_cnn_aape_top5_test, n=672
        )

        moving_average_total = moving_average(aape_total, n=672)
        moving_average_aape_validation = moving_average(aape_validation, n=672)
        moving_average_aape_test = moving_average(aape_test, n=672)

        hybrid_moving_average_total = moving_average(hybrid_aape_total, n=672)
        hybrid_moving_average_aape_validation = moving_average(
            hybrid_aape_validation, n=672
        )
        hybrid_moving_average_aape_test = moving_average(hybrid_aape_test, n=672)

        load_factor_moving_average = moving_average(load_factor, n=672)

        moving_average_aape_baseline_total = [None] * 4
        moving_average_aape_baseline_validation = [None] * 4
        moving_average_aape_baseline_test = [None] * 4

        for i in range(4):
            moving_average_aape_baseline_total[i] = moving_average(
                aape_baseline_total[i], n=672
            )
            moving_average_aape_baseline_validation[i] = moving_average(
                aape_baseline_validation[i], n=672
            )
            moving_average_aape_baseline_test[i] = moving_average(
                aape_baseline_test[i], n=672
            )

        padding = np.empty(672)
        padding[:] = np.nan

        x_valid = np.arange(len(padding) + len(moving_average_aape_top5_validation))
        x_test = np.arange(len(padding) + len(moving_average_aape_top5_test)) + len(
            x_valid
        )
        x_total = np.arange(len(x_valid) + len(x_test))
        x_middle = x_total[len(x_valid) : len(x_valid) + 672]

        print(len(x_total))
        print(len(x_valid) + len(x_middle) + len(x_test))
        print(len(timestamps_total_datetime))

        # xt_valid = timestamps_total_datetime[np.arange(x_valid)]
        # xt_middle = timestamps_total_datetime[len(x_valid):len(x_valid)+672]
        xt_test = timestamps_total_datetime[-len(target) : :]

        moving_average_top5_middle = moving_average_top5_total[
            len(x_valid) - 672 : len(x_valid)
        ]
        hybrid_moving_average_top5_middle = hybrid_moving_average_top5_total[
            len(x_valid) - 672 : len(x_valid)
        ]
        hybrid_from_cnn_moving_average_top5_middle = hybrid_from_cnn_moving_average_top5_total[
            len(x_valid) - 672 : len(x_valid)
        ]
        moving_average_baseline_middle = [None] * 4
        for i in range(4):
            moving_average_baseline_middle[i] = moving_average_aape_baseline_total[i][
                len(x_valid) - 672 : len(x_valid)
            ]
        moving_average_middle = moving_average_total[
            :, len(x_valid) - 672 : len(x_valid)
        ]
        hybrid_moving_average_middle = hybrid_moving_average_total[
            :, len(x_valid) - 672 : len(x_valid)
        ]

        print(len(moving_average_aape_top5_validation))
        print(len(hybrid_moving_average_aape_top5_validation))

        # fig = plt.figure()
        # sns.distplot(aape_top5_test, label='top 5')
        # sns.distplot(aape_baseline_test[0], label='LightGBM')
        # sns.distplot(aape_test[rank[0]], label='valid rank 0')
        # sns.distplot(aape_top5_test-aape_baseline_test[0][0][hybrid_adjustment::])
        # print(np.mean(aape_top5_test-aape_baseline_test[0][0][hybrid_adjustment::]))
        # print(scipy.stats.sem(aape_top5_test-aape_baseline_test[0][0][hybrid_adjustment::]))
        # plt.show()
        # plt.close(fig)

        # print(len(aape_top5_test))
        # print(len(aape_baseline_test[0][0]))
        _baseline_id = 0
        t, p = scipy.stats.ttest_rel(
            aape_top5_test, aape_baseline_test[_baseline_id][0]
        )
        print(
            "{:>16} ttest: top5                 mean: {:>8.3f}, baseline mean: {:>8.3f}, diff: {:>8.3f}, p-value: {:>8.3f}".format(
                region,
                np.mean(aape_top5_test),
                np.mean(aape_baseline_test[_baseline_id][0]),
                np.mean(aape_top5_test - aape_baseline_test[_baseline_id][0]),
                p,
            )
        )
        t, p = scipy.stats.ttest_rel(
            hybrid_aape_top5_test, aape_baseline_test[_baseline_id][0]
        )
        print(
            "{:>16} ttest: top5 hybrid          mean: {:>8.3f}, baseline mean: {:>8.3f}, diff: {:>8.3f}, p-value: {:>8.3f}".format(
                region,
                np.mean(hybrid_aape_top5_test),
                np.mean(aape_baseline_test[_baseline_id][0]),
                np.mean(hybrid_aape_top5_test - aape_baseline_test[_baseline_id][0]),
                p,
            )
        )
        t, p = scipy.stats.ttest_rel(
            hybrid_from_cnn_aape_top5_test, aape_baseline_test[_baseline_id][0]
        )
        print(
            "{:>16} ttest: top5 hybrid from cnn mean: {:>8.3f}, baseline mean: {:>8.3f}, diff: {:>8.3f}, p-value: {:>8.3f}".format(
                region,
                np.mean(hybrid_from_cnn_aape_top5_test),
                np.mean(aape_baseline_test[_baseline_id][0]),
                np.mean(
                    hybrid_from_cnn_aape_top5_test - aape_baseline_test[_baseline_id][0]
                ),
                p,
            )
        )

        a = calculate_maape(estimated, target)
        b = calculate_maape(np.expand_dims(estimated_top5_test, 0), target)[0]
        t, p = scipy.stats.ttest_1samp(a, b)
        print(
            "{:>16} ttest: top5                maape: {:>8.3f},     CNN maape: {:>8.3f}, p-value: {:>8.3f}".format(
                region, b, np.mean(a), p
            )
        )
        print()

        # _baseline_id = 0
        # t, p = scipy.stats.ttest_rel(hybrid_from_cnn_aape_top5_test, aape_baseline_test[_baseline_id][0])
        # if (t < 0 and p < 0.05):
        #     print(f"{bcolors.OKGREEN}########  hybrid ensemble er bedre enn LightGBM ##########{bcolors.ENDC}")
        # else:
        #     print(f"{bcolors.FAIL}?????????{bcolors.ENDC}")

        # _baseline_id = 1
        # t, p = scipy.stats.ttest_rel(hybrid_from_cnn_aape_top5_test, aape_baseline_test[_baseline_id][0])
        # if (t < 0 and p < 0.05):
        #     print(f"{bcolors.OKGREEN}########  hybrid ensemble er bedre enn Random Forest ##########{bcolors.ENDC}")
        # else:
        #     print(f"{bcolors.FAIL}?????????{bcolors.ENDC}")

        # _baseline_id = 2
        # t, p = scipy.stats.ttest_rel(hybrid_from_cnn_aape_top5_test, aape_baseline_test[_baseline_id][0])
        # if (t < 0 and p < 0.05):
        #     print(f"{bcolors.OKGREEN}########  hybrid ensemble er bedre enn kNN ##########{bcolors.ENDC}")
        # else:
        #     print(f"{bcolors.FAIL}?????????{bcolors.ENDC}")

        # _baseline_id = 3
        # t, p = scipy.stats.ttest_rel(hybrid_from_cnn_aape_top5_test, aape_baseline_test[_baseline_id][0])
        # if (t < 0 and p < 0.05):
        #     print(f"{bcolors.OKGREEN}########  hybrid ensemble er bedre enn AdaBoost ##########{bcolors.ENDC}")
        # else:
        #     print(f"{bcolors.FAIL}?????????{bcolors.ENDC}")

        print()
        print()

        try:
            tmp = pd.DataFrame(columns=["id", "aape", "time", "type", "model"])
            for i, series in enumerate(moving_average_total):
                a = len(series)
                b = len(timestamps_total_datetime)

                d = {
                    "id": rank[i],
                    "aape": series,
                    "time": timestamps_total_datetime[(b - a) : :],
                    "type": "total",
                    "model": "CNN models",
                }
                df = pd.DataFrame(data=d)
                tmp = tmp.append(df, ignore_index=True)

            for i, series in enumerate(hybrid_moving_average_total):
                a = len(series)
                b = len(timestamps_total_datetime)
                d = {
                    "id": hybrid_rank[i],
                    "aape": series,
                    "time": timestamps_total_datetime[(b - a) : :],
                    "type": "total",
                    "model": "Hybrid models",
                }
                df = pd.DataFrame(data=d)
                tmp = tmp.append(df, ignore_index=True)

            a = len(moving_average_top5_total)
            b = len(timestamps_total_datetime)

            d = {
                "id": 0,
                "aape": moving_average_top5_total,
                "time": timestamps_total_datetime[(b - a) : :],
                "type": "total",
                "model": "CNN top 5\nensemble",
            }
            df = pd.DataFrame(data=d)
            tmp = tmp.append(df, ignore_index=True)

            # d = {
            #     'id': 0,
            #     'aape': hybrid_moving_average_top5_total,
            #     'time': timestamps_total_datetime[671::],
            #     'type': 'total',
            #     'model': 'Hybrid top 5'
            # }
            # df = pd.DataFrame(data=d)
            # tmp = tmp.append(df, ignore_index=True)
            a = len(hybrid_from_cnn_moving_average_top5_total)
            b = len(timestamps_total_datetime)
            d = {
                "id": 0,
                "aape": hybrid_from_cnn_moving_average_top5_total,
                "time": timestamps_total_datetime[(b - a) : :],
                "type": "total",
                "model": "Hybrid top 5\nensemble",
            }
            df = pd.DataFrame(data=d)
            tmp = tmp.append(df, ignore_index=True)

            # d = {
            #     'id': 0,
            #     'aape': moving_average_aape_baseline_total[0],
            #     'time': timestamps_total_datetime[671::],
            #     'type': 'total',
            #     'model': 'Baseline (LightGBM)'
            # }
            # df = pd.DataFrame(data=d)
            # tmp = tmp.append(df, ignore_index=True)

            # d = {
            #     'id': 0,
            #     'aape': moving_average_aape_baseline_total[1],
            #     'time': timestamps_total_datetime[671::],
            #     'type': 'total',
            #     'model': 'baseline (RF)'
            # }
            # df = pd.DataFrame(data=d)
            # tmp = tmp.append(df, ignore_index=True)

            # fig = plt.figure(figsize=(5,4))
            fig = plt.figure(figsize=(8, 6))
            with palette_3:
                sns.lineplot(x="time", y="aape", hue="model", ci="sd", data=tmp)
            plt.legend(loc="upper left")
            plt.title(
                "1 month running mean ({}) of the validaion and test set - {}".format(
                    _metric.upper(), region
                )
            )
            plt.xlabel("Date (yyyy-mm)")
            plt.ylabel("{}".format(short[_metric.upper()]))
            min_aape = min(
                np.min(moving_average_aape_validation), np.min(moving_average_aape_test)
            )
            max_aape = max(
                np.max(moving_average_aape_validation), np.max(moving_average_aape_test)
            )
            plt.axvline(timestamps_total_datetime[len(valid_target)], 0.05, 0.95, c="k")
            # plt.show()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                "out/1_month_running_mean_({})_of_the_validation_set_including_hybrids-{}.pdf".format(
                    _metric.upper(), region
                )
            )
            # plt.show()

            plt.close(fig)

        except:
            print(
                f"{bcolors.FAIL}[Info] Failed plot (hybrid running mean{bcolors.ENDC}"
            )

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        flag = True
        for i, series in enumerate(moving_average_aape_validation):
            if i == rank[0]:
                ax1.plot(
                    x_valid,
                    np.append(padding, series),
                    lw=1,
                    c="C1",
                    label="validation rank 1",
                )
            elif flag:
                ax1.plot(
                    x_valid,
                    np.append(padding, series),
                    lw=0.5,
                    c="k",
                    alpha=0.2,
                    label="CNN models",
                )
                flag = False
            else:
                ax1.plot(x_valid, np.append(padding, series), lw=0.5, c="k", alpha=0.2)

        for i, series in enumerate(moving_average_middle):
            if i == rank[0]:
                ax1.plot(x_middle, series, "--", lw=1, alpha=0.5, c="C1")
            else:
                ax1.plot(x_middle, series, lw=0.5, c="k", alpha=0.1)

        for i, series in enumerate(moving_average_aape_test):
            if i == rank[0]:
                ax1.plot(x_test, np.append(padding, series), lw=1, c="C1")
            else:
                ax1.plot(x_test, np.append(padding, series), lw=0.5, c="k", alpha=0.2)

        # flag = True
        # for i, series in enumerate(hybrid_moving_average_aape_validation):
        #     if i == rank[0] and False:
        #         plt.plot(x_valid, np.append(padding, series), lw=1, c='C1', label="validation rank 1 (hybrid)")
        #     elif flag:
        #         plt.plot(x_valid, np.append(padding, series), lw=0.5, c="b", alpha=0.3, label="Hybrid models")
        #         flag = False
        #     else:
        #         plt.plot(x_valid, np.append(padding, series), lw=0.5, c="b", alpha=0.3)

        # for i, series in enumerate(hybrid_moving_average_middle):
        #     if i == rank[0] and False:
        #         plt.plot(x_middle, series, '--', lw=1, alpha=0.5, c='C1')
        #     else:
        #         plt.plot(x_middle, series, lw=0.5, c="b", alpha=0.3)

        # for i, series in enumerate(hybrid_moving_average_aape_test):
        #     if i == rank[0] and False:
        #         plt.plot(x_test, np.append(padding, series), lw=1, c='C1')
        #     else:
        #         plt.plot(x_test, np.append(padding, series), lw=0.5, c="b", alpha=0.3)

        ax1.plot(
            np.arange(len(target_total_baseline[0])),
            np.append(padding, moving_average_aape_baseline_total[0][1::]),
            lw=1,
            c="C6",
            label="baseline (LightGBM)",
        )
        ax1.plot(
            np.arange(len(target_total_baseline[1])),
            np.append(padding, moving_average_aape_baseline_total[1][1::]),
            lw=1,
            c="C5",
            label="baseline (RF)",
        )

        # plt.plot(np.arange(len(target_total_baseline[2])), np.append(padding, moving_average_aape_baseline_total[2][1::]), lw=1, c='C4', label='baseline (kNN)')
        # plt.plot(np.arange(len(target_total_baseline[3])), np.append(padding, moving_average_aape_baseline_total[3][1::]), lw=1, c='C6', label='baseline (AdaBoost)')

        # plt.plot()

        ax1.plot(
            x_valid,
            np.append(padding, moving_average_aape_top5_validation),
            c="C0",
            lw=1.5,
            label="top 5 ensemble",
        )
        ax1.plot(
            x_test, np.append(padding, moving_average_aape_top5_test), c="C0", lw=1.5
        )
        ax1.plot(x_middle, moving_average_top5_middle, "--", lw=1, c="C0", alpha=0.5)

        # plt.plot(x_valid, np.append(padding, hybrid_moving_average_aape_top5_validation), c='C1', lw=1.5, label="hybrid top 5")
        # plt.plot(x_test, np.append(padding, hybrid_moving_average_aape_top5_test), c='C1', lw=1.5)
        # plt.plot(x_middle, hybrid_moving_average_top5_middle, '--', lw=1, c='C1', alpha=0.5)

        # plt.plot(x_valid, np.append(padding, hybrid_from_cnn_moving_average_aape_top5_validation), c='C3', lw=1.5, label="hybrid top 5 from cnn")
        # plt.plot(x_test, np.append(padding, hybrid_from_cnn_moving_average_aape_top5_test), c='C3', lw=1.5)
        # plt.plot(x_middle, hybrid_from_cnn_moving_average_top5_middle, '--', lw=1, c='C3', alpha=0.5)

        # ax1.legend(loc="upper right")
        ax1.legend(loc="best")
        min_aape = min(
            np.min(moving_average_aape_validation), np.min(moving_average_aape_test)
        )
        max_aape = max(
            np.max(moving_average_aape_validation), np.max(moving_average_aape_test)
        )
        ax1.vlines(len(valid_target), min_aape, max_aape)

        ax1.set_ylabel(_metric.upper())
        ax2.set_xlabel("Date")
        ax1.set_title(
            "1 month running mean ({}) of the validaion and test set - {}".format(
                _metric.upper(), region
            )
        )

        ax2.set_ylabel("Target\nload factor")
        ax2.plot(
            x_total[-len(load_factor_moving_average) : :],
            load_factor_moving_average,
            c="k",
            lw=1,
        )
        print(len(load_factor_moving_average))
        print(len(x_total))
        print(load_factor_moving_average[-1])
        # ax = plt.gca()
        if _metric == "me":
            ax1.axhline(color="k", ls=":", lw=1)

        xticks = ax2.get_xticks()
        ax2.set_xticklabels(
            [timestamps_total_labels[int(i)] for i in xticks[0 : len(xticks) - 1]]
        )

        figname = "out/moving_average_{}-{}.pdf".format(_metric.upper(), region)
        plt.savefig(figname)
        fig.tight_layout()
        # plt.show()
        plt.close(fig)

        # subprocess.run(['krop', '--autotrim', '--go', figname])

        estimated_top5_test_other = np.loadtxt(
            "out/top5-test-{}{}{}.csv".format(region, other_version, h)
        )
        estimated_top5_test_other = estimated_top5_test_other[:, 1]
        aape_top5_test_other = metric_partial(
            np.expand_dims(estimated_top5_test_other, 0), target
        )
        aape_top5_test_other = aape_top5_test_other[0]

        print(f"{bcolors.WARNING}{estimated_top5_test_other.shape}{bcolors.ENDC}")

        # fig=plt.figure()

        fig, ax1 = plt.subplots(figsize=(5, 4))
        plt.title("Production in region {}".format(region))
        f = h5.File("dataset/{}.h5".format(region), "r")
        capacity = f["target/capacity"][:]
        capacity_target = capacity[-len(target) : :]
        f.close()

        ax1.plot(
            xt_test,
            np.divide(estimated_baseline[0][0], capacity_target),
            "--",
            c="C3",
            label="baseline (LightGBM)",
        )
        ax1.plot(
            xt_test,
            np.divide(estimated_baseline[1][0], capacity_target),
            "--",
            c="C4",
            label="baseline (RF)",
        )

        if version == "4":
            ax1.plot(
                xt_test,
                np.divide(estimated_top5_test, capacity_target),
                lw=1.5,
                c="C1",
                label="top 5 ensemble\n(ordinal classification)",
            )
            # ax1.plot(xt_test, np.divide(estimated_top5_test_other, capacity_target), lw=1.5, c='C0', label='top 5 ensemble\n(single node regression)')
        elif version == "5":
            ax1.plot(
                xt_test,
                np.divide(estimated_top5_test_other, capacity_target),
                lw=1.5,
                c="C0",
                label="top 5 ensemble\n(ordinal classification)",
            )
            ax1.plot(
                xt_test,
                np.divide(estimated_top5_test, capacity_target),
                lw=1.5,
                c="C1",
                label="top 5 ensemble\n(single node regression)",
            )
        else:
            print("[Error] is version correct? [4, 5]")
            exit()
        ax1.plot(
            xt_test, np.divide(target, capacity_target), c="C0", lw=1, label="Target"
        )
        # plt.plot(estimated_baseline[2][0], '--', alpha=0.5, label='baseline (kNN)')
        # plt.plot(estimated_baseline[3][0], '--', alpha=0.5, label='baseline (AdaBoost)')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Load Factor")
        # ax= plt.gca()
        # xticks = ax.get_xticks()
        # ax.set_xticklabels([timestamps_test_labels[int(i)] for i in xticks[0:len(xticks)-1]])
        fig.autofmt_xdate()

        ax2 = ax1.twinx()
        ax2.set_ylabel("Production Volume - MWh/h")
        ax2.plot(xt_test, target, c="C0", alpha=0, lw=1)
        ax2.grid(False)
        fig.tight_layout()

        # ax1.legend(loc='upper left')
        # plt.show()
        plt.close(fig)

        target_sorted_args = np.argsort(target)

        q1 = np.max(target) * 1 / 5
        q2 = np.max(target) * 2 / 5
        q3 = np.max(target) * 3 / 5
        q4 = np.max(target) * 4 / 5

        print(q1, q2, q3, q4)

        l1, l2, l3, l4, l5 = ([], [], [], [], [])
        for i, v in enumerate(target):
            if v < q1:
                l1.append(i)
            elif v < q2:
                l2.append(i)
            elif v < q3:
                l3.append(i)
            elif v < q4:
                l4.append(i)
            else:
                l5.append(i)

        idxs = [np.array(l1), np.array(l2), np.array(l3), np.array(l4), np.array(l5)]

        tmp = pd.DataFrame(columns=["aape", "model", "load factor"])
        # idxs = np.array_split(np.arange(len(target)), 5)

        # target_load_factor = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
        target_load_factor = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        try:
            with open("out/scores-ord-snr.pk", "rb") as f:
                scores_ord_snr = pickle.load(f)
                # time.sleep(1)

            with open("out/scores-ord-lgbm.pk", "rb") as f:
                scores_ord_lgbm = pickle.load(f)
                # time.sleep(1)

            with open("out/scores-snr-lgbm.pk", "rb") as f:
                scores_snr_lgbm = pickle.load(f)
                # time.sleep(1)
        except:
            print("init varialbes")
            scores_ord_snr = np.zeros((5, 3))
            scores_ord_lgbm = np.zeros((5, 3))
            scores_snr_lgbm = np.zeros((5, 3))

        for i, idx in enumerate(idxs):
            lf = target_load_factor[i]
            if version == "4":
                d = {
                    "aape": aape_top5_test[idx],
                    "model": "top 5 ensemble\n(ordinal classification)",
                    "load factor": lf,
                }
                df = pd.DataFrame(data=d)
                tmp = tmp.append(df, ignore_index=True)

                d = {
                    "aape": aape_top5_test_other[idx],
                    "model": "top 5 ensemble\n(single node regression)",
                    "load factor": lf,
                }
                df = pd.DataFrame(data=d)
                tmp = tmp.append(df, ignore_index=True)
            if version == "5":
                d = {
                    "aape": aape_top5_test[idx],
                    "model": "top 5 ensemble\n(single node regression)",
                    "load factor": lf,
                }
                df = pd.DataFrame(data=d)
                tmp = tmp.append(df, ignore_index=True)

                d = {
                    "aape": aape_top5_test_other[idx],
                    "model": "top 5 ensemble\n(ordinal classification)",
                    "load factor": lf,
                }
                df = pd.DataFrame(data=d)
                tmp = tmp.append(df, ignore_index=True)

            d = {
                "aape": aape_baseline_test[0][0][::][idx],
                "model": "baseline (LightGBM)",
                "load factor": lf,
            }
            df = pd.DataFrame(data=d)
            tmp = tmp.append(df, ignore_index=True)

            d = {
                "aape": aape_baseline_test[1][0][::][idx],
                "model": "baseline (RF)",
                "load factor": lf,
            }
            df = pd.DataFrame(data=d)
            tmp = tmp.append(df, ignore_index=True)

            d = {
                "aape": aape_baseline_test[2][0][::][idx],
                "model": "baseline (kNN)",
                "load factor": lf,
            }
            df = pd.DataFrame(data=d)
            tmp = tmp.append(df, ignore_index=True)

            d = {
                "aape": aape_baseline_test[3][0][::][idx],
                "model": "baseline (AdaBoost)",
                "load factor": lf,
            }
            df = pd.DataFrame(data=d)
            tmp = tmp.append(df, ignore_index=True)

            print()

            # assume main is ordinal and other is snr!!!!! aka --comment 'asdf.4.4'

            t, p = scipy.stats.ttest_rel(
                aape_top5_test[idx], aape_baseline_test[0][0][::][idx]
            )
            print(
                "{}: top5 mean aape ord: {:>8.3f}, lgbm mean aape: {:>8.3f}, p={:>8.3f}".format(
                    lf,
                    np.mean(aape_top5_test[idx]),
                    np.mean(aape_baseline_test[0][0][::][idx]),
                    p,
                )
            )

            if p > 0.05:
                scores_ord_lgbm[i, 2] += 1
            elif t > 0:
                scores_ord_lgbm[i, 1] += 1
            else:
                scores_ord_lgbm[i, 0] += 1

            # t, p = scipy.stats.ttest_rel(aape_top5_test_other[idx], aape_baseline_test[0][0][::][idx])
            # print('{}: top5 mean aape snr: {:>8.3f}, lgbm mean aape: {:>8.3f}, p={:>8.3f}'.format(lf, np.mean(aape_top5_test_other[idx]), np.mean(aape_baseline_test[0][0][::][idx]), p))

            # if p > 0.05:
            #     scores_snr_lgbm[i, 2] += 1
            # elif t > 0:
            #     scores_snr_lgbm[i, 1] += 1
            # else:
            #     scores_snr_lgbm[i, 0] += 1

            # t, p = scipy.stats.ttest_rel(aape_top5_test[idx], aape_top5_test_other[idx])
            # print('{}: top5 mean aape ord: {:>8.3f}, snr mean aape: {:>8.3f}, p={:>8.3f}'.format(lf, np.mean(aape_top5_test[idx]), np.mean(aape_top5_test_other[idx]), p))

            # if p > 0.05:
            #     scores_ord_snr[i, 2] += 1
            # elif t > 0:
            #     scores_ord_snr[i, 1] += 1
            # else:
            #     scores_ord_snr[i, 0] += 1

        with open("out/scores-ord-snr.pk", "wb") as f:
            pickle.dump(scores_ord_snr, f)
            # time.sleep(1)
        with open("out/scores-ord-lgbm.pk", "wb") as f:
            pickle.dump(scores_ord_lgbm, f)
            # time.sleep(1)
        with open("out/scores-snr-lgbm.pk", "wb") as f:
            pickle.dump(scores_snr_lgbm, f)
            # time.sleep(1)

        fig = plt.figure()
        sns.boxplot(
            x="load factor", y="aape", hue="model", fliersize=2, linewidth=1, data=tmp
        )
        plt.ylabel("{}".format(short[_metric.upper()]))
        plt.xlabel("Load Factor (%)")
        plt.title(
            "Distribution of model accuracy based on target load factor - {}".format(
                region
            )
        )
        plt.savefig(
            "out/distribution_of_model_accuracy_({})_based_on_target_load_factor-{}.pdf".format(
                _metric.upper(), region
            )
        )

        # g1 = gp.gnuplotlib(title='aape', _with='points')
        # line1 = (target[target_sorted_args], aape_top5_test[target_sorted_args], {'_with': 'points'})
        # line2 = (target[target_sorted_args], aape_baseline_test[3][0][hybrid_adjustment::][target_sorted_args], {'_with': 'points'})
        # gp.plot(line1, line2)


def autocorrelation():
    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK2",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]
    regions = regions_ger + regions_nrd

    targets = []

    for region in regions:
        f = h5.File("dataset/{}.h5".format(region), "r")
        target = f["target/production"][:]
        target = target[~np.isnan(target)]
        targets.append(target)
        f.close()
    for i, region in enumerate(regions):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        plot_acf(targets[i], ax=ax1, lags=48, c="C0")
        ax1.set_title("Autocorrelation - {}".format(region))
        # ax1.set_xlabel("lag - hours")
        ax1.set_ylabel("correlation")

        plot_pacf(targets[i], ax=ax2, lags=48, c="C0")
        ax2.set_title("Partial Autocorrelation - {}".format(region))
        ax2.set_xlabel("correlation lag in hours")
        ax2.set_ylabel("correlation")
        ax1.grid(True)
        ax2.grid(True)

        f.align_ylabels([ax1, ax2])
        plt.savefig("out/partial-autocorrelation-{}.pdf".format(region))
        plt.close()


def eval_deu(db, param, verbose=0):
    print()
    print("aggregated statistics for germany")
    regions = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    eval_multi_region(db, param, regions, "DEU", verbose)


def eval_nrd(db, param, verbose=0):
    print()
    print("aggregated statistics for the nordics")
    regions = [
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]
    eval_multi_region(db, param, regions, "NRD", verbose)


def eval_multi_region(db, param, regions, region_global, verbose=0):
    if verbose >= 2:
        print("regions included:", regions)
        print("filter:")
        pprint(param)
        print()

    targets = {}
    timestamps = {}
    idx = {}
    estimations = {}

    # read from database
    # for region in regions:
    #     param['region'] = region
    #     targets[region] = db.get(**param, target=True)
    #     timestamps[region] = db.get(**param, is_timestamps=True)
    #     estimations[region] = db.get(**param, target=False)

    # read from top5 files
    # for region in regions:
    #     data = np.loadtxt(f'out/top5-test-{region}4.csv')
    #     data = data.T
    #     targets[region] = data[2]
    #     timestamps[region] = data[0]
    #     estimations[region] = data[1]
    for region in regions:
        data = np.loadtxt(f"out/test-{region}5.csv")
        data = data.T
        targets[region] = data[-1]
        timestamps[region] = data[0]
        estimations[region] = data[1:-1]

    # only one target. pick first one
    for region in regions:
        if len(targets[region].shape) == 2:
            targets[region] = targets[region][0]
            if verbose > 1:
                print("warning: multiple targets for {}".format(region))

    if verbose > 0:
        print("\nestimations")
        for key, value in estimations.items():
            print(key, value.shape)
    if verbose > 1:
        print("\ntargets")
        for key, value in targets.items():
            print(key, value.shape)
        print("\ntimestamps")
        for key, value in timestamps.items():
            print(key, value.shape)

    # an extremly complicated way to find the intersection of the timestamps
    common_timestamps = list(map(set, timestamps.values()))
    common_timestamps = set.intersection(*common_timestamps)
    common_timestamps = np.array(list(common_timestamps))

    if verbose > 0:
        print("\ncommon timestamps:", len(common_timestamps))

    for region in regions:
        index_orig = np.arange(len(timestamps[region]))
        idx[region] = index_orig[np.isin(timestamps[region], common_timestamps)]

    # only relevant if i compare a set of models for each region
    total_production = np.zeros(len(common_timestamps))
    total_estimation = np.zeros((10000, len(common_timestamps)))

    for region in regions:
        total_production += targets[region][idx[region]]
        for i in range(10000):
            index = np.random.choice(len(estimations[region]))
            total_estimation[i] += estimations[region][index][idx[region]]

    # total_production = np.zeros(len(common_timestamps))
    # total_estimation = np.zeros((1, len(common_timestamps)))

    # for region in regions:
    #     total_production += targets[region][idx[region]]
    #     total_estimation[0] += estimations[region][idx[region]]
    print_stats(total_estimation, total_production, region_global)


def print_stats(estimations, target, region=None):
    maape = calculate_maape(estimations, target)
    mape = calculate_mape(estimations, target, region=region)
    mae = calculate_mae(estimations, target)
    me = calculate_me(estimations, target)

    # sns.distplot(mape)
    # plt.show()

    if estimations.shape[0] == 1:
        print("[Info] standard deviation of the mean does not apply to only one series")
        print("MAAPE mean: {:>8.3f}".format(np.mean(maape)))
        print("MAPE  mean: {:>8.3f}".format(np.mean(mape)))
        print("MAE   mean: {:>8.3f}".format(np.mean(mae)))
        print("ME    mean: {:>8.3f}".format(np.mean(me)))
        return

    print("MAAPE: mean: {:.4}".format(np.mean(maape)))
    print("        std: {:.4}".format(np.std(maape)))
    print("       best: {:.4}".format(np.min(maape)))
    print("      worst: {:.4}".format(np.max(maape)))
    print("        sem: {:.4}".format(3 * scipy.stats.sem(maape)))
    print("      2*std: {:.4}".format(2 * np.std(maape)))
    print()
    print("MAPE:  mean: {:.4}".format(np.mean(mape)))
    print("        std: {:.4}".format(np.std(mape)))
    print("       best: {:.4}".format(np.min(mape)))
    print("      worst: {:.4}".format(np.max(mape)))
    print("        sem: {:.4}".format(3 * scipy.stats.sem(mape)))
    print("      2*std: {:.4}".format(2 * np.std(mape)))
    print()
    print("MAE:   mean: {:.5}".format(np.mean(mae)))
    print("        std: {:.5}".format(np.std(mae)))
    print("       best: {:.5}".format(mae[np.argmin(np.abs(mae))]))
    print("      worst: {:.5}".format(mae[np.argmax(np.abs(mae))]))
    print("        sem: {:.4}".format(3 * scipy.stats.sem(mae)))
    print("      2*std: {:.4}".format(2 * np.std(mae)))
    print()
    print("ME:    mean: {:.5}".format(np.mean(me)))
    print("        std: {:.5}".format(np.std(me)))
    print("       best: {:.5}".format(me[np.argmin(np.abs(me))]))
    print("      worst: {:.5}".format(me[np.argmax(np.abs(me))]))
    print("        sem: {:.4}".format(3 * scipy.stats.sem(me)))
    print("      2*std: {:.4}".format(2 * np.std(me)))
    print()

    confidence = 3
    print(
        "MAAPE 0.997 confidence interval (SEM): ({:.4}, {:.4}),\tmean={:.4}".format(
            np.mean(maape) - confidence * scipy.stats.sem(maape),
            np.mean(maape) + confidence * scipy.stats.sem(maape),
            np.mean(maape),
        )
    )
    print(
        "MAPE  0.997 confidence interval (SEM): ({:.4}, {:.4}),\tmean={:.4}".format(
            np.mean(mape) - confidence * scipy.stats.sem(mape),
            np.mean(mape) + confidence * scipy.stats.sem(mape),
            np.mean(mape),
        )
    )
    print(
        "MAE   0.997 confidence interval (SEM): ({:.5}, {:.5}),\tmean={:.5}".format(
            np.mean(mae) - confidence * scipy.stats.sem(mae),
            np.mean(mae) + confidence * scipy.stats.sem(mae),
            np.mean(mae),
        )
    )
    print(
        "ME    0.997 confidence interval (SEM): ({:.5}, {:.5}),\tmean={:.5}".format(
            np.mean(me) - confidence * scipy.stats.sem(me),
            np.mean(me) + confidence * scipy.stats.sem(me),
            np.mean(me),
        )
    )


def strip_args(args):
    d = vars(args)
    minimal_d = {}

    for key, value in d.items():
        if value is not None:
            minimal_d[key] = value
    minimal_d.pop("verbose", None)

    if "data_length" in minimal_d.keys() and "window_size" in minimal_d.keys():
        minimal_d["data_length"] -= 2 * minimal_d["window_size"]

    if "timestamp_start" in minimal_d.keys():
        minimal_d["timestamp_start"] = get_utc(minimal_d["timestamp_start"])

    return minimal_d


def check_4_5_agains_baselines(db, param, verbose):
    ID = 3
    metrics = pd.DataFrame(columns=["maape", "mape", "mae", "me", "region", "model"])

    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK2",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]

    regions = regions_ger + regions_nrd

    targets = {}
    timestamps = {}
    idx = {}
    estimations = {}

    strategies = ["4.4", "4.5", "4.6"]

    strategies_map = {
        "4.4": "Ordinal Classification",
        "4.5": "Single Node Regression",
        "4.6": "Baseline (LightGBM)",
    }

    try:
        metrics = pickle.load(open("out/metrics_4562.pk", "rb"))
    except:
        for region in regions:
            for s in strategies:
                c = f"{region}.{s}"
                param["region"] = region
                param["comment"] = c
                if verbose >= 2:
                    print("[Debug] fetching: {}".format(c))
                targets[c] = db.get(**param, target=True)
                timestamps[c] = db.get(**param, is_timestamps=True)
                estimations[c] = db.get(**param, target=False)

        # only one target. pick first one
        for region in regions:
            for s in strategies:
                c = f"{region}.{s}"
                if len(targets[c].shape) == 2:
                    targets[c] = targets[c][0]
                    if verbose > 1:
                        print("warning: multiple targets for {}".format(c))
                if len(estimations[c].shape) == 1:
                    estimations[c] = np.expand_dims(estimations[c], axis=0)
                if region in regions_ger:
                    estimations[c] *= 1  # 1.05
                if region in regions_nrd:
                    estimations[c] *= 1  # 1.03

        for region in regions:
            instances = -1
            for i, s in enumerate(strategies):
                c = f"{region}.{s}"
                if estimations[c].shape[0] > instances:
                    instances = estimations[c].shape[0]
            maape = [None] * len(strategies)
            mape = [None] * len(strategies)
            mae = [None] * len(strategies)
            me = [None] * len(strategies)

            for i, ns in enumerate(strategies):
                c = f"{region}.{ns}"
                if estimations[c].shape[1] == 0:
                    continue
                maape[i] = calculate_maape(estimations[c], targets[c])
                mape[i] = calculate_mape(estimations[c], targets[c], region=region)
                mae[i] = calculate_mae(estimations[c], targets[c])
                me[i] = calculate_me(estimations[c], targets[c])

            for i, s in enumerate(strategies):
                d = {
                    "maape": maape[i],
                    "mape": mape[i],
                    "mae": mae[i],
                    "me": me[i],
                    "region": region,
                    "model": strategies_map[s],
                }
                # print(d)
                df = pd.DataFrame(data=d)
                metrics = metrics.append(df, ignore_index=True)
        pickle.dump(metrics, open("out/metrics_456.pk", "wb"))

    if args.figures:
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = sns.boxplot(
            x="region",
            y="maape",
            hue="model",
            fliersize=2,
            linewidth=1,
            data=metrics[
                metrics["model"].isin(
                    ["Single Node Regression", "Ordinal Classification"]
                )
            ],
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.yaxis.grid(True)
        ax.xaxis.grid(linestyle="--")
        # plt.grid(True)
        plt.ylabel("Metric - MAAPE")
        plt.xlabel("Region")
        plt.title(
            "Error distribution comparing Single Node Regression and Ordinal Classification"
        )
        plt.tight_layout()
        figname = "out/single_node_regression_vs_ordinal_classification-maape.pdf"
        plt.savefig(figname)

        if verbose >= 1:
            print("[Info] saved: ", figname)
        # subprocess.run(['krop', '--autotrim', '--go', figname])
        if verbose >= 2:
            print("[Debug] fig cropped")
        plt.close(fig)

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = sns.boxplot(
            x="region",
            y="mape",
            hue="model",
            fliersize=2,
            linewidth=1,
            data=metrics[
                metrics["model"].isin(
                    ["Single Node Regression", "Ordinal Classification"]
                )
            ],
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.yaxis.grid(True)
        ax.xaxis.grid(linestyle="--")
        # plt.grid(True)
        plt.ylabel("Metric - MAPE")
        plt.xlabel("Region")
        plt.title(
            "Error distribution comparing Single Node Regression and Ordinal Classification"
        )
        plt.tight_layout()
        figname = "out/single_node_regression_vs_ordinal_classification-mape.pdf"
        plt.savefig(figname)
        if verbose >= 1:
            print("[Info] saved: ", figname)
        # subprocess.run(['krop', '--autotrim', '--go', figname])
        if verbose >= 2:
            print("[Debug] fig cropped")
        plt.close(fig)

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.gca()
        sns.boxplot(
            x="region",
            y="mae",
            hue="model",
            ax=ax,
            fliersize=2,
            linewidth=1,
            data=metrics[
                metrics["model"].isin(
                    ["Single Node Regression", "Ordinal Classification"]
                )
            ],
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        # ax.set_yscale('log')
        ax.yaxis.grid(True)
        ax.xaxis.grid(linestyle="--")
        plt.ylabel("Metric - MAE")
        plt.xlabel("Region")
        # plt.grid(True)
        plt.title(
            "Error distribution comparing Single Node Regression and Ordinal Classification"
        )
        plt.tight_layout()
        figname = "out/single_node_regression_vs_ordinal_classification-mae.pdf"
        plt.savefig(figname)
        if verbose >= 1:
            print("[Info] saved: ", figname)
        # subprocess.run(['krop', '--autotrim', '--go', figname])
        if verbose >= 2:
            print("[Debug] fig cropped")
        plt.close(fig)

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.gca()
        sns.boxplot(
            x="region",
            y="me",
            hue="model",
            ax=ax,
            fliersize=2,
            linewidth=1,
            data=metrics[
                metrics["model"].isin(
                    ["Single Node Regression", "Ordinal Classification"]
                )
            ],
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        # ax.set_yscale('symlog')
        ax.yaxis.grid(True)
        ax.xaxis.grid(linestyle="--")
        # plt.grid(True)
        plt.ylabel("Metric - ME")
        plt.xlabel("Region")
        plt.title(
            "Error distribution comparing Single Node Regression and Ordinal Classification"
        )
        plt.tight_layout()
        figname = "out/single_node_regression_vs_ordinal_classification-me.pdf"
        plt.savefig(figname)
        if verbose >= 1:
            print("[Info] saved: ", figname)
        # subprocess.run(['krop', '--autotrim', '--go', figname])
        if verbose >= 2:
            print("[Debug] fig cropped")

        plt.close(fig)

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.gca()
        sns.boxplot(
            x="region",
            y="mae",
            hue="model",
            ax=ax,
            fliersize=2,
            linewidth=1,
            data=metrics[
                metrics["model"].isin(
                    ["Single Node Regression", "Ordinal Classification"]
                )
            ],
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.set_yscale("log")
        ax.yaxis.grid(True)
        ax.xaxis.grid(linestyle="--")
        plt.ylabel("Metric - MAE")
        plt.xlabel("Region")
        # plt.grid(True)
        plt.title(
            "Error distribution comparing Single Node Regression and Ordinal Classification"
        )
        plt.tight_layout()
        figname = (
            "out/single_node_regression_vs_ordinal_classification-mae-logarithmic.pdf"
        )
        plt.savefig(figname)
        if verbose >= 1:
            print("[Info] saved: ", figname)
        # subprocess.run(['krop', '--autotrim', '--go', figname])
        if verbose >= 2:
            print("[Debug] fig cropped")
        plt.close(fig)

        fig = plt.figure(figsize=(6.5, 6.5))
        ax = fig.gca()
        sns.boxplot(
            x="region",
            y="me",
            hue="model",
            ax=ax,
            fliersize=2,
            linewidth=1,
            data=metrics[
                metrics["model"].isin(
                    ["Single Node Regression", "Ordinal Classification"]
                )
            ],
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        )
        ax.set_yscale("symlog")
        ax.yaxis.grid(True)
        ax.xaxis.grid(linestyle="--")
        # plt.grid(True)
        plt.ylabel("Metric - ME")
        plt.xlabel("Region")
        plt.title(
            "Error distribution comparing Single Node Regression and Ordinal Classification"
        )
        plt.tight_layout()
        figname = (
            "out/single_node_regression_vs_ordinal_classification-me-logarithmic.pdf"
        )
        plt.savefig(figname)
        if verbose >= 1:
            print("[Info] saved: ", figname)
        # subprocess.run(['krop', '--autotrim', '--go', figname])
        if verbose >= 2:
            print("[Debug] fig cropped")

        plt.close(fig)

    print(
        "{:>16}: {:>16} {:>16}         {:>8} {:>8}".format(
            "region", "ordinal", "single", "diff", "p-value"
        )
    )

    for region in regions:
        # print()
        # print("from:", get_utc(timestamps[f'{region}.4.4'][0]))
        # print("to..:", get_utc(timestamps[f'{region}.4.4'][-1]))

        maape_ordinal = metrics.loc[metrics["model"] == "Ordinal Classification"].loc[
            metrics["region"] == region
        ][args.rank_metric.lower()]
        maape_single = metrics.loc[metrics["model"] == "Single Node Regression"].loc[
            metrics["region"] == region
        ][args.rank_metric.lower()]

        t, p = scipy.stats.ttest_ind(maape_ordinal, maape_single)
        twostd_ordinal = 2 * scipy.stats.sem(maape_ordinal)
        twostd_single = 2 * scipy.stats.sem(maape_single)

        twostd_ordinal = 2 * np.std(maape_ordinal)
        twostd_single = 2 * np.std(maape_single)

        if args.latex:
            print(
                "{:>16} & \\num{{{:>8.3f} \\pm {:<8.3f}}} & \\num{{{:>8.3f} \\pm {:<8.3f}}} & \\num{{{:>8.3f}}} & \\num{{{:>8.3f}}} \\\\".format(
                    region,
                    np.mean(maape_ordinal),
                    twostd_ordinal,
                    np.mean(maape_single),
                    twostd_single,
                    np.mean(maape_ordinal) - np.mean(maape_single),
                    p,
                )
            )
        else:
            print(
                "{:>16}: {:>8.3f} +/- {:<8.3f} {:>8.3f} +/- {:<8.3f} {:>8.3f} {:>8.3f}".format(
                    region,
                    np.mean(maape_ordinal),
                    twostd_ordinal,
                    np.mean(maape_single),
                    twostd_single,
                    np.mean(maape_ordinal) - np.mean(maape_single),
                    p,
                )
            )


def test_ns(db, param, verbose):
    ID = 3
    metrics = pd.DataFrame(columns=["maape", "mape", "mae", "me", "region", "NS"])

    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK2",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]

    regions = regions_ger + regions_nrd

    targets = {}
    timestamps = {}
    idx = {}
    estimations = {}
    strategies = ["NS1", "NS2", "NS3", "NS4"]

    try:
        metrics = pickle.load(open("out/metrics_normalization.pk", "rb"))
    except:
        for region in regions:
            for ns in strategies:
                c = f"{region}.{ID}.{ns}"
                param["region"] = region
                param["comment"] = c
                if verbose >= 2:
                    print("[Debug] fetching: {}".format(c))
                targets[c] = db.get(**param, target=True)
                timestamps[c] = db.get(**param, is_timestamps=True)
                estimations[c] = db.get(**param, target=False)

        # only one target. pick first one
        for region in regions:
            for ns in strategies:
                c = f"{region}.{ID}.{ns}"
                if len(targets[c].shape) == 2:
                    targets[c] = targets[c][0]
                    if verbose > 1:
                        print("warning: multiple targets for {}".format(c))
                # estimations[c] = np.expand_dims(estimations[c], axis=0)
                if region in regions_ger:
                    estimations[c] *= 1  # 1.05
                if region in regions_nrd:
                    estimations[c] *= 1  # 1.03

        for region in regions:
            instances = -1
            for i, ns in enumerate(strategies):
                c = f"{region}.{ID}.{ns}"
                if estimations[c].shape[0] > instances:
                    instances = estimations[c].shape[0]
            maape = [None] * 4
            mape = [None] * 4
            mae = [None] * 4
            me = [None] * 4

            for i, ns in enumerate(strategies):
                c = f"{region}.{ID}.{ns}"
                if estimations[c].shape[1] == 0:
                    continue
                maape[i] = calculate_maape(estimations[c], targets[c])
                mape[i] = calculate_mape(estimations[c], targets[c], region=region)
                mae[i] = calculate_mae(estimations[c], targets[c])
                me[i] = calculate_me(estimations[c], targets[c])

            for i in range(4):
                d = {
                    "maape": maape[i],
                    "mape": mape[i],
                    "mae": mae[i],
                    "me": me[i],
                    "region": region,
                    "NS": strategies[i],
                }
                df = pd.DataFrame(data=d)
                metrics = metrics.append(df, ignore_index=True)
        pickle.dump(metrics, open("out/metrics_normalization.pk", "wb"))

    print(metrics.describe())
    t_values_ns12 = np.empty(shape=(4, len(regions)))
    p_values_ns12 = np.empty(shape=(4, len(regions)))
    t_values_ns12[:] = np.nan
    p_values_ns12[:] = np.nan

    t_values_ns34 = np.empty(shape=(4, len(regions)))
    p_values_ns34 = np.empty(shape=(4, len(regions)))
    t_values_ns34[:] = np.nan
    p_values_ns34[:] = np.nan

    global_best_mean_std = 0
    local_best_mean_std = 0
    no_difference_mean_std = 0

    global_best_min_max = 0
    local_best_min_max = 0
    no_difference_min_max = 0

    print(
        "{:}{:>16} {:>8} {:>8} {:>8} {:>8}{:}".format(
            bcolors.HEADER,
            "region",
            "global",
            "local",
            "t-value",
            "p-value",
            bcolors.ENDC,
        )
    )
    x = np.arange(len(regions))
    for i, m in enumerate(["maape", "mape", "mae", "me"]):
        for j, region in enumerate(regions):
            filter1 = metrics["region"] == region
            filter2 = metrics["NS"] == "NS1"
            maape1 = metrics.where(filter1 & filter2)[m].dropna().to_numpy()
            filter2 = metrics["NS"] == "NS2"
            maape2 = metrics.where(filter1 & filter2)[m].dropna().to_numpy()

            t, p = scipy.stats.ttest_ind(maape1, maape2, equal_var=False)
            t_values_ns12[i, j] = t
            p_values_ns12[i, j] = p
            if m == "maape":
                if t < 0 and p < 0.05:
                    print(
                        "{:>16} {:}{:8.3f}{:} {:8.3f} {:8.3f} {:8.3f}".format(
                            region,
                            bcolors.WARNING,
                            np.mean(maape1),
                            bcolors.ENDC,
                            np.mean(maape2),
                            t,
                            p,
                        )
                    )
                    global_best_mean_std += 1
                elif t > 0 and p < 0.05:
                    print(
                        "{:>16} {:8.3f} {:}{:8.3f}{:} {:8.3f} {:8.3f}".format(
                            region,
                            np.mean(maape1),
                            bcolors.WARNING,
                            np.mean(maape2),
                            bcolors.ENDC,
                            t,
                            p,
                        )
                    )
                    local_best_mean_std += 1
                else:
                    print(
                        "{:>16} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
                            region, np.mean(maape1), np.mean(maape2), t, p
                        )
                    )
                    no_difference_mean_std += 1

            filter2 = metrics["NS"] == "NS3"
            maape1 = metrics.where(filter1 & filter2)[m].dropna().to_numpy()
            filter2 = metrics["NS"] == "NS4"
            maape2 = metrics.where(filter1 & filter2)[m].dropna().to_numpy()

            t, p = scipy.stats.ttest_ind(maape1, maape2, equal_var=False)
            t_values_ns34[i, j] = t
            p_values_ns34[i, j] = p
            if m == "maape":
                if t < 0 and p < 0.05:
                    print(
                        "{:>16} {:}{:8.3f}{:} {:8.3f} {:8.3f} {:8.3f}".format(
                            "mm " + region,
                            bcolors.WARNING,
                            np.mean(maape1),
                            bcolors.ENDC,
                            np.mean(maape2),
                            t,
                            p,
                        )
                    )
                    global_best_min_max += 1
                elif t > 0 and p < 0.05:
                    print(
                        "{:>16} {:8.3f} {:}{:8.3f}{:} {:8.3f} {:8.3f}".format(
                            "mm " + region,
                            np.mean(maape1),
                            bcolors.WARNING,
                            np.mean(maape2),
                            bcolors.ENDC,
                            t,
                            p,
                        )
                    )
                    local_best_min_max += 1
                else:
                    print(
                        "{:>16} {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
                            "mm " + region, np.mean(maape1), np.mean(maape2), t, p
                        )
                    )
                    no_difference_min_max += 1

    print("global_best_mean_std..:", global_best_mean_std)
    print("local_best_mean_std...:", local_best_mean_std)
    print("no_difference_mean_std:", no_difference_mean_std)
    print("global_best_min_max..:", global_best_min_max)
    print("local_best_min_max...:", local_best_min_max)
    print("no_difference_min_max:", no_difference_min_max)

    # g1 = gp.gnuplotlib(title = 't-values',
    #                    _with = 'points')
    # g1.plot((x, t_values_ns12[0]), (x, t_values_ns34[0]))

    sns.distplot(t_values_ns12[0], rug=True)
    sns.distplot(t_values_ns34[0], rug=True)
    plt.show()
    # np.savetxt("out/regions.csv", regions, fmt="%s")
    # np.savetxt("out/t_values_ns13.csv", t_values_ns12.T)
    # np.savetxt("out/t_values_ns24.csv", t_values_ns34.T)

    if args.figures:
        terminal = 'pdfcairo solid color font ",12" size 4.8in,5in'
        g = gp.gnuplotlib(
            terminal=terminal,
            hardcopy="output.pdf",
            multiplot="layout 4,1 title 'Comparison between the local and global normalization strategies'",
        )

        curve_options_2l = dict(_with="line lt 3 lc 0")
        curve2l = (np.array([0, len(x)]), np.array([2, 2]), curve_options_2l)
        curvem2l = (np.array([0, len(x)]), np.array([-2, -2]), curve_options_2l)

        curve_options0 = dict(_with="points pt 7 lc 2")
        curve_options1 = dict(_with="points pt 9 lc 1")
        curve0 = (x, t_values_ns12[0], curve_options0)
        curve1 = (x, t_values_ns34[0], curve_options1)
        subplot_options0 = dict(
            unset="xtics", set="yrange [-3.5:3.5]", title="MAAPE", ylabel="t-value"
        )
        subplot0 = (curve0, curve1, curve2l, curvem2l, subplot_options0)

        curve_options1 = dict(_with="points pt 7 lc 2")
        curve_options2 = dict(_with="points pt 9 lc 1")
        curve1 = (x, t_values_ns12[1], curve_options1)
        curve2 = (x, t_values_ns34[1], curve_options2)
        subplot_options1 = dict(
            unset="xtics", set="yrange [-3.5:3.5]", title="MAPE*", ylabel="t-value"
        )
        subplot1 = (curve1, curve2, curve2l, curvem2l, subplot_options1)

        curve_options3 = dict(_with="points pt 7 lc 2")
        curve_options4 = dict(_with="points pt 9 lc 1")
        curve3 = (x, t_values_ns12[2], curve_options3)
        curve4 = (x, t_values_ns34[2], curve_options4)
        subplot_options2 = dict(
            unset="xtics", set="yrange [-3.5:3.5]", title="MAE", ylabel="t-value"
        )
        subplot2 = (curve3, curve4, curve2l, curvem2l, subplot_options2)

        curve_options5 = dict(_with="points pt 7 lc 2")
        curve_options6 = dict(_with="points pt 9 lc 1")
        curve5 = (x, t_values_ns12[3], curve_options5)
        curve6 = (x, t_values_ns34[3], curve_options6)
        subplot_options3 = dict(
            unset="xtics", set="yrange [-3.5:3.5]", title="ME", ylabel="t-value"
        )
        subplot3 = (curve5, curve6, curve2l, curvem2l, subplot_options3)

        g.plot(subplot0, subplot1, subplot2, subplot3)

    # gp.plot(np.concatenate((t_values_ns12[0].flatten(), t_values_ns34[0].flatten())), histogram=True, binwidth=0.5)

    # fig = plt.figure()
    # sns.boxplot(x='NS', y='maape', data=metrics)
    # plt.show()

    m = ["maape", "mape", "mae", "me"]
    for t in [-2, 2]:
        for i, tmp in enumerate(m):
            if t == 2:
                print(
                    "13 {} t={}".format(tmp, t),
                    scipy.stats.ttest_1samp(t_values_ns12[i], t),
                )
                print(
                    "24 {} t={}".format(tmp, t),
                    scipy.stats.ttest_1samp(t_values_ns34[i], t),
                )
            if t == -2:
                print(
                    "13 {} t={}".format(tmp, t),
                    scipy.stats.ttest_1samp(-t_values_ns12[i], -t),
                )
                print(
                    "24 {} t={}".format(tmp, t),
                    scipy.stats.ttest_1samp(-t_values_ns34[i], -t),
                )


def misc(db, param, verbose=0):
    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK2",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]
    regions = regions_nrd + regions_ger

    for region in regions:
        f = h5.File(f"dataset/{region}.h5", "r")
        ratio = f["target/ratio"][:]
        ratio = ratio[~np.isnan(ratio)]
        transformed_ratio = -np.multiply(ratio, ratio) + 2 * ratio
        f.close()

        np.savetxt(
            f"out/transformation_{region}.csv", np.stack((ratio, transformed_ratio)).T
        )


def ttest(db, param, verbose=0):
    print(f"ttest for {args.region1} and {args.region2}")

    # param['region'] = param['region1']
    # param['comment'] = param['comment1']
    region1 = param["comment"].split(".")[0]
    region2 = param["comment2"].split(".")[0]
    target1 = db.get(**param, target=True)
    timestamps1 = db.get(**param, is_timestamps=True)
    if len(target1.shape) > 1:
        print("Warning. target burde ikke ha mer enn 1 rad")
        target1 = target1[0]
    estimated1 = db.get(**param, target=False)

    # param['region'] = param['region2']
    param["comment"] = param["comment2"]
    target2 = db.get(**param, target=True)
    timestamps2 = db.get(**param, is_timestamps=True)
    if len(target2.shape) > 1:
        print("Warning. target burde ikke ha mer enn 1 rad")
        target2 = target2[0]
    estimated2 = db.get(**param, target=False)

    if len(target2) == 0 or len(target1) == 0:
        print("No targets found")
        return
    if len(estimated2) == 0 or len(estimated1) == 0:
        print("No estimates found")
        return

    maape1 = calculate_maape(estimated1, target1)
    mape1 = calculate_mape(estimated1, target1, region=region1)
    mae1 = calculate_mae(estimated1, target1)
    me1 = calculate_me(estimated1, target1)

    maape2 = calculate_maape(estimated2, target2)
    mape2 = calculate_mape(estimated2, target2, region=region2)
    mae2 = calculate_mae(estimated2, target2)
    me2 = calculate_me(estimated2, target2)

    maape_tvalue, maape_pvalue = scipy.stats.ttest_ind(maape1, maape2, equal_var=False)
    mape_tvalue, mape_pvalue = scipy.stats.ttest_ind(mape1, mape2, equal_var=False)
    mae_tvalue, mae_pvalue = scipy.stats.ttest_ind(mae1, mae2, equal_var=False)
    me_tvalue, me_pvalue = scipy.stats.ttest_ind(me1, me2, equal_var=False)

    # if the p-value is less than 0.05 then the distribution means are different
    # with statistical significance

    # print(f"number of samples in region {args.region1}, comment {args.comment1}: {len(maape1)}")
    print(
        f"5 best maape: {np.round(np.sort(maape1), 3)[0:5]}, with ids: {np.argsort(maape1)[0:5]}"
    )
    print(
        f"5 best mape.: {np.round(np.sort(mape1), 3)[0:5]}, with ids: {np.argsort(mape1)[0:5]}"
    )
    print(
        f"5 best mae..: {np.round(np.sort(mae1), 3)[0:5]}, with ids: {np.argsort(mae1)[0:5]}"
    )
    print(
        f"5 best me...: {np.round(me1[np.argsort(np.abs(me1))], 3)[0:5]}, with ids: {np.argsort(np.abs(me1))[0:5]}"
    )

    print()
    # print(f"number of samples in region {args.region2}, comment {args.comment2}: {len(maape2)}")
    print(
        f"5 best maape: {np.round(np.sort(maape2), 3)[0:5]}, with ids: {np.argsort(maape2)[0:5]}"
    )
    print(
        f"5 best mape.: {np.round(np.sort(mape2), 3)[0:5]}, with ids: {np.argsort(mape2)[0:5]}"
    )
    print(
        f"5 best mae..: {np.round(np.sort(mae2), 3)[0:5]}, with ids: {np.argsort(mae2)[0:5]}"
    )
    print(
        f"5 best me...: {np.round(me2[np.argsort(np.abs(me2))], 3)[0:5]}, with ids: {np.argsort(np.abs(me2))[0:5]}"
    )
    print()
    # print(f"regions:\t{param['region1']}\t\t{param['region2']}")
    # print(f"comment:\t{param['comment1']}\t{param['comment2']}")
    print(
        "  maape:\t{:8.3f}\t{:8.3f}".format(
            np.around(np.mean(maape1), 3), np.around(np.mean(maape2), 3)
        )
    )
    print(
        "  mape.:\t{:8.3f}\t{:8.3f}".format(
            np.around(np.mean(mape1), 3), np.around(np.mean(mape2), 3)
        )
    )
    print(
        "  mae..:\t{:8.3f}\t{:8.3f}".format(
            np.around(np.mean(mae1), 3), np.around(np.mean(mae2), 3)
        )
    )
    print(
        "  me...:\t{:8.3f}\t{:8.3f}".format(
            np.around(np.mean(me1), 3), np.around(np.mean(me2), 3)
        )
    )
    print()
    print("comment:\tt-value\t\tp-value")
    print(
        "  maape:\t{:8.3f}\t{:8.3f}".format(
            np.around(maape_tvalue, 3), np.around(maape_pvalue, 3)
        )
    )
    print(
        "  mape.:\t{:8.3f}\t{:8.3f}".format(
            np.around(mape_tvalue, 3), np.around(mape_pvalue, 3)
        )
    )
    print(
        "  mae..:\t{:8.3f}\t{:8.3f}".format(
            np.around(mae_tvalue, 3), np.around(mae_pvalue, 3)
        )
    )
    print(
        "  me...:\t{:8.3f}\t{:8.3f}".format(
            np.around(me_tvalue, 3), np.around(me_pvalue, 3)
        )
    )


def measure_baseline_performance(metrics):
    maape = {}
    maape_nt = {}
    mape = {}
    mape_nt = {}
    mae = {}
    mae_nt = {}
    me = {}
    me_nt = {}

    t_values = np.empty(shape=(4, 4))
    p_values = np.empty(shape=(4, 4))
    t_values[:] = np.nan
    p_values[:] = np.nan

    models = ["LightGBM", "Random Forest", "kNN", "AdaBoost"]
    for model in models:
        filter1 = metrics["Transformed Ratio"] == True
        filter2 = metrics["model"] == model
        maape[model] = metrics.where(filter1 & filter2)["maape"].dropna().to_numpy()
        maape_nt[model] = metrics.where(~filter1 & filter2)["maape"].dropna().to_numpy()
        mape[model] = metrics.where(filter1 & filter2)["mape"].dropna().to_numpy()
        mape_nt[model] = metrics.where(~filter1 & filter2)["mape"].dropna().to_numpy()
        mae[model] = metrics.where(filter1 & filter2)["mae"].dropna().to_numpy()
        mae_nt[model] = metrics.where(~filter1 & filter2)["mae"].dropna().to_numpy()
        me[model] = metrics.where(filter1 & filter2)["me"].dropna().to_numpy()
        me_nt[model] = metrics.where(~filter1 & filter2)["me"].dropna().to_numpy()

    print("paired sampeled t-test of the different baseline models.")
    formatter = {"float_kind": lambda x: "{:15.3f}".format(x)}

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if model1 == model2:
                continue
            t, p = scipy.stats.ttest_rel(np.abs(maape[model1]), np.abs(maape[model2]))
            t_values[i, j] = t
            p_values[i, j] = p

    print("MAAPE")
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("t-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(t_values[i], formatter=formatter)
            )
        )
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("p-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(p_values[i], formatter=formatter)
            )
        )
    print()

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if model1 == model2:
                continue
            t, p = scipy.stats.ttest_rel(np.abs(mape[model1]), np.abs(mape[model2]))
            t_values[i, j] = t
            p_values[i, j] = p

    print("MAPE")
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("t-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(t_values[i], formatter=formatter)
            )
        )
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("p-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(p_values[i], formatter=formatter)
            )
        )
    print()

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if model1 == model2:
                continue
            t, p = scipy.stats.ttest_rel(np.abs(mae[model1]), np.abs(mae[model2]))
            t_values[i, j] = t
            p_values[i, j] = p

    print("MAE")
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("t-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(t_values[i], formatter=formatter)
            )
        )
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("p-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(p_values[i], formatter=formatter)
            )
        )
    print()

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if model1 == model2:
                continue
            t, p = scipy.stats.ttest_rel(np.abs(me[model1]), np.abs(me[model2]))
            t_values[i, j] = t
            p_values[i, j] = p

    print("ME")
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("t-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(t_values[i], formatter=formatter)
            )
        )
    print("{:16} {:>16}{:>16}{:>16}{:>16}".format("p-value", *models))
    for i, model in enumerate(models):
        print(
            "{:>16} {:}".format(
                model, np.array2string(p_values[i], formatter=formatter)
            )
        )
    print()

    filter1 = metrics["Transformed Ratio"] == True
    filter2 = metrics["model"] == "LightGBM"
    filter3 = metrics["model"] == "Random Forest"
    maape_lgbm = metrics.where(filter1 & filter2)["maape"].dropna().to_numpy()
    maape_rf = metrics.where(filter1 & filter3)["maape"].dropna().to_numpy()

    print(np.mean(maape_lgbm), np.mean(maape_rf))
    print(scipy.stats.ttest_rel(maape_lgbm, maape_rf))


def measure_transformation_effect(metrics, model):
    filter1 = metrics["Transformed Ratio"] == True
    filter2 = metrics["model"] == model
    maape = metrics.where(filter1 & filter2)["maape"].dropna().to_numpy()
    maape_nt = metrics.where(~filter1 & filter2)["maape"].dropna().to_numpy()
    mape = metrics.where(filter1 & filter2)["mape"].dropna().to_numpy()
    mape_nt = metrics.where(~filter1 & filter2)["mape"].dropna().to_numpy()
    mae = metrics.where(filter1 & filter2)["mae"].dropna().to_numpy()
    mae_nt = metrics.where(~filter1 & filter2)["mae"].dropna().to_numpy()
    me = metrics.where(filter1 & filter2)["me"].dropna().to_numpy()
    me_nt = metrics.where(~filter1 & filter2)["me"].dropna().to_numpy()
    t_maape, p_maape = scipy.stats.ttest_rel(maape, maape_nt)
    t_mape, p_mape = scipy.stats.ttest_rel(mape, mape_nt)
    t_mae, p_mae = scipy.stats.ttest_rel(mae, mae_nt)
    t_me, p_me = scipy.stats.ttest_rel(np.abs(me), np.abs(me_nt))

    print()
    print(
        f"A paired sample t-test is performed to measure the effect of ratio transformation using {model} baseline."
    )
    print("negative t value means the transformation improved the results\n")
    print("       {:8} {:8}".format("t-value", "p-value"))
    print(
        "maape: {:>8.3f} {:>8.3f}, mean with tr,: {:>8.3f}, mean without tr.: {:>8.3f}, difference: {:>8.3f}".format(
            t_maape,
            p_maape,
            np.mean(maape),
            np.mean(maape_nt),
            np.mean(maape - maape_nt),
        )
    )
    print(
        "mape.: {:>8.3f} {:>8.3f}, mean with tr,: {:>8.3f}, mean without tr.: {:>8.3f}, difference: {:>8.3f}".format(
            t_mape, p_mape, np.mean(mape), np.mean(mape_nt), np.mean(mape - mape_nt)
        )
    )
    print(
        "mae..: {:>8.3f} {:>8.3f}, mean with tr,: {:>8.3f}, mean without tr.: {:>8.3f}, difference: {:>8.3f}".format(
            t_mae, p_mae, np.mean(mae), np.mean(mae_nt), np.mean(mae - mae_nt)
        )
    )
    print(
        "me...: {:>8.3f} {:>8.3f}, mean with tr,: {:>8.3f}, mean without tr.: {:>8.3f}, difference: {:>8.3f}".format(
            t_me, p_me, np.mean(me), np.mean(me_nt), np.mean(me - me_nt)
        )
    )


def eval_baselines_wrapper(db, param, verbose):
    ID = 4
    metrics = pd.DataFrame(
        columns=["maape", "mape", "mae", "me", "Transformed Ratio", "region", "model"]
    )
    # print(f"{bcolors.BOLD}NO HARMONICS{bcolors.ENDC}")
    # print("\n--------------------- LightGBM ----------------------")
    # eval_baselines(db, param, verbose, '3', ID)
    # print("\n------------------ random forest --------------------")
    # eval_baselines(db, param, verbose, '7', ID)
    # print("\n----------------------- kNN -------------------------")
    # eval_baselines(db, param, verbose, '9', ID)
    # print("\n--------------------- AdaBoost ----------------------")
    # eval_baselines(db, param, verbose, 'b', ID)

    print(f"\n\n{bcolors.BOLD}WITH HARMONICS{bcolors.ENDC}")
    print("\n--------------------- LightGBM ----------------------")
    metrics = eval_baselines(db, param, verbose, "6", ID, metrics)
    print("\n------------------ random forest --------------------")
    metrics = eval_baselines(db, param, verbose, "8", ID, metrics)
    print("\n----------------------- kNN -------------------------")
    metrics = eval_baselines(db, param, verbose, "a", ID, metrics)
    print("\n--------------------- AdaBoost ----------------------")
    metrics = eval_baselines(db, param, verbose, "c", ID, metrics)

    measure_transformation_effect(metrics, "LightGBM")
    measure_transformation_effect(metrics, "Random Forest")
    measure_transformation_effect(metrics, "kNN")
    measure_transformation_effect(metrics, "AdaBoost")

    measure_baseline_performance(metrics)

    # 9000D1, '#049F75'

    if args.figures:
        with palette_normal:
            fig = plt.figure(figsize=(5, 5))
            sns.boxplot(
                x="model",
                y="maape",
                hue="Transformed Ratio",
                fliersize=2,
                linewidth=1,
                data=metrics,
            )
            plt.title("Baseline comparison across all regions - MAAPE")
            plt.ylabel("MAAPE")
            plt.legend(loc="upper left", title="Transform")
            plt.gca().yaxis.grid(True)
            plt.tight_layout()
            plt.savefig("out/baselines-boxplot-maape.pdf")
            plt.close(fig)
            fig = plt.figure(figsize=(5, 5))
            sns.boxplot(
                x="model",
                y="mape",
                hue="Transformed Ratio",
                fliersize=2,
                linewidth=1,
                data=metrics,
            )
            plt.title("Baseline comparison across all regions - MAPE*")
            plt.ylabel("MAPE*")
            plt.legend(loc="upper left", title="Transform")
            plt.gca().yaxis.grid(True)
            plt.tight_layout()
            plt.savefig("out/baselines-boxplot-mape.pdf")
            plt.close(fig)
            fig = plt.figure(figsize=(5, 5))
            sns.boxplot(
                x="model",
                y="mae",
                hue="Transformed Ratio",
                fliersize=2,
                linewidth=1,
                data=metrics,
            )
            plt.title("Baseline comparison across all regions - MAE")
            plt.ylabel("MAE")
            plt.legend(loc="upper left", title="Transform")
            plt.gca().yaxis.grid(True)
            plt.savefig("out/baselines-boxplot-mae.pdf")
            plt.tight_layout()
            plt.close(fig)
            fig = plt.figure(figsize=(5, 5))
            sns.boxplot(
                x="model",
                y="me",
                hue="Transformed Ratio",
                fliersize=2,
                linewidth=1,
                data=metrics,
            )
            plt.title("Baseline comparison across all regions - ME")
            plt.ylabel("ME")
            plt.legend(loc="upper left", title="Transform")
            plt.gca().yaxis.grid(True)
            plt.ylim(-500, 200)
            plt.tight_layout()
            plt.savefig("out/baselines-boxplot-me.pdf")
            plt.close(fig)

    # f, axes = plt.subplots(2, 2, figsize=(11,10))
    # ax1 = sns.boxplot(x='model', y='maape', hue='Transformed Ratio', data=metrics, palette="Set3", ax=axes[0, 0])
    # ax2 = sns.boxplot(x='model', y='mape', hue='Transformed Ratio', data=metrics, palette="Set3", ax=axes[0, 1])
    # ax3 = sns.boxplot(x='model', y='mae', hue='Transformed Ratio', data=metrics, palette="Set3", ax=axes[1, 0])
    # ax4 = sns.boxplot(x='model', y='me', hue='Transformed Ratio', data=metrics, palette="Set3", ax=axes[1, 1])

    # axes[0,0].get_legend().remove()
    # axes[0,1].get_legend().remove()
    # axes[1,0].get_legend().remove()
    # axes[1,1].get_legend().remove()

    # handles, labels = ax4.get_legend_handles_labels()

    # f.suptitle("Baseline comparison across all regions.")
    # plt.show()
    # exit()


def select_comments(db, param, comment, ID, regions):
    targets = {}
    timestamps = {}
    estimations = {}

    for region in regions:
        c = f"{region}.{ID}.{region}"
        c_nt = f"{region}.{ID}.{region}.nt"
        param["region"] = region
        param["comment"] = c
        targets[c] = db.get(**param, target=True)
        timestamps[c] = db.get(**param, is_timestamps=True)
        estimations[c] = db.get(**param, target=False)


def eval_baselines(db, param, verbose, comment, ID, metrics):
    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK2",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]
    regions = regions_ger + regions_nrd

    targets = {}
    timestamps = {}
    idx = {}
    estimations = {}

    models = {
        "3": "LightGBM",
        "6": "LightGBM",
        "7": "Random Forest",
        "8": "Random Forest",
        "9": "kNN",
        "a": "kNN",
        "b": "AdaBoost",
        "c": "AdaBoost",
    }

    # comment selection
    # selected_comments_ger = select_comments(db, param, comment, ID, regions_ger)
    # selected_comments_nrd = select_comments(db, param, comment, ID, regions_nrd)

    for region in regions:
        for tr in ["", ".nt"]:
            c = f"{region}.{ID}.{comment}{tr}"
            param["region"] = region
            param["comment"] = c
            targets[c] = db.get(**param, target=True)
            timestamps[c] = db.get(**param, is_timestamps=True)
            estimations[c] = db.get(**param, target=False)

    # only one target. pick first one
    for region in regions:
        for tr in ["", ".nt"]:
            c = f"{region}.{ID}.{comment}{tr}"
            if len(targets[c].shape) == 2:
                targets[c] = targets[c][0]
                if verbose > 1:
                    print("warning: multiple targets for {}".format(c))
            estimations[c] = np.expand_dims(estimations[c], axis=0)
            if region in regions_ger:
                estimations[c] *= 1  # 1.05
            if region in regions_nrd:
                estimations[c] *= 1  # 1.03

    best_regions_ger = []
    best_regions_nrd = []

    if args.latex:
        s = """
    \\begin{table}[htb]
    \\centering
    \\begin{tabular}{lrrrr}
    \\hline"""
        print(s)
        print(
            "\\textbf{region} & \\textbf{maape}  & \\textbf{mape} & \\textbf{mae} & \\textbf{me} \\\\"
        )
        print("\\hline")
    else:
        print(
            "{:}{:>16}: {:8} {:8} {:8} {:8}{:}".format(
                bcolors.BOLD, "region", "maape", "mape", "mae", "me", bcolors.ENDC
            )
        )
    for region in regions:
        maape = np.empty(2)
        mape = np.empty(2)
        mae = np.empty(2)
        me = np.empty(2)

        maape[:] = np.nan
        mape[:] = np.nan
        mae[:] = np.nan
        me[:] = np.nan

        # TODO:
        # if '+' in region:
        #     e = None
        #     t = None
        #     region_split = region.split('+')
        #     for r in region_split:
        #         c = f"{r}.{ID}.{comment}{tr}"
        #         print(c)
        #         if estimations[c].shape[1] == 0:
        #             continue
        #         if e is None:
        #             e = np.zeros_like(estimations[c])
        #             t = np.zeros_like(targets[c])
        #         print(e.shape)
        #         if "DK1off" in c:
        #             e[0:3163] += estimations[c][0][0:3163]
        #             t[0:3163] += targets[c][0][0:3163]
        #         else:
        #             e += estimations[c]
        #             t += targets[c]

        #     maape[i] = calculate_maape(e, t)[0]
        #     mape[i] = calculate_mape(e, t)[0]
        #     mae[i] = calculate_mae(e, t)[0]
        #     me[i] = calculate_me(e, t)[0]

        for i, tr in enumerate(["", ".nt"]):
            c = f"{region}.{ID}.{comment}{tr}"
            if estimations[c].shape[1] == 0:
                continue
            maape[i] = calculate_maape(estimations[c], targets[c])[0]
            mape[i] = calculate_mape(estimations[c], targets[c], region=region)[0]
            mae[i] = calculate_mae(estimations[c], targets[c])[0]
            me[i] = calculate_me(estimations[c], targets[c])[0]

        d = {
            "maape": maape,
            "mape": mape,
            "mae": mae,
            "me": me,
            "Transformed Ratio": [True, False],
            "region": [region, region],
            "model": [models[comment], models[comment]],
        }
        df = pd.DataFrame(data=d)
        metrics = metrics.append(df, ignore_index=True)

        scores = np.zeros(2)
        scores += np.array([4, 0])[np.argsort(maape)]
        scores += np.array([2, 0])[np.argsort(mape)]
        scores += np.array([0.5, 0])[np.argsort(mae)]
        scores += np.array([1, 0])[np.argsort(np.abs(me))]

        ranks = np.argsort(scores)

        best = ""
        if ranks[0] > ranks[1]:
            best = ": (yes)"
        elif np.abs(ranks[0] - ranks[1]) < 0.001:
            best = ": (eqal)"
        else:
            best = ""
        if args.latex:
            print(
                "{:>16} & {:8.3f} & {:8.3f} & {:8.3f} & {:8.3f} \\\\".format(
                    region, maape[0], mape[0], mae[0], me[0]
                )
            )
        else:
            print(
                "{:>16}: {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
                    region, maape[0], mape[0], mae[0], me[0]
                )
            )
        if args.include_nt:
            if ranks[1] > ranks[0]:
                best = f": {bcolors.WARNING}(yes){bcolors.ENDC}"
            elif np.abs(ranks[0] - ranks[1]) < 0.001:
                best = ""
            else:
                best = ""
            if args.latex:
                print(
                    "{:>13} nt & {:8.3f} & {:8.3f} & {:8.3f} & {:8.3f} \\\\".format(
                        region, maape[1], mape[1], mae[1], me[1]
                    )
                )
            else:
                print(
                    "{:>13} nt: {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
                        region, maape[1], mape[1], mae[1], me[1]
                    )
                )

        if "+" not in region and region != "DK1" and region != "DK2":
            if ranks[1] > ranks[0]:
                if region in regions_ger:
                    best_regions_ger.append(f"{region}.{ID}.{comment}.nt")
                if region in regions_nrd:
                    best_regions_nrd.append(f"{region}.{ID}.{comment}.nt")
            else:
                if region in regions_ger:
                    best_regions_ger.append(f"{region}.{ID}.{comment}")
                if region in regions_nrd:
                    best_regions_nrd.append(f"{region}.{ID}.{comment}")
        # if '+' in region:
        #    regions.remove(region)

    if args.latex:
        s = """\\hline
    \\end{tabular}
    \\caption[]{}
    \\label{}
    \\end{table}"""
        print(s)
    if verbose >= 2:
        print("[Debug] best regions:", best_regions_ger)
        print("[Debug] best regions:", best_regions_nrd)

    timestamps_ger = {}
    targets_ger = {}
    estimations_ger = {}

    timestamps_nrd = {}
    targets_nrd = {}
    estimations_nrd = {}

    for c in best_regions_ger:
        timestamps_ger[c] = timestamps[c]
        targets_ger[c] = targets[c]
        estimations_ger[c] = estimations[c]

    for c in best_regions_nrd:
        timestamps_nrd[c] = timestamps[c]
        targets_nrd[c] = targets[c]
        estimations_nrd[c] = estimations[c]

    # an extremly complicated way to find the intersection of the timestamps
    common_timestamps_ger = list(map(set, timestamps_ger.values()))
    common_timestamps_ger = set.intersection(*common_timestamps_ger)
    common_timestamps_ger = np.array(list(common_timestamps_ger))

    # an extremly complicated way to find the intersection of the timestamps
    common_timestamps_nrd = list(map(set, timestamps_nrd.values()))
    common_timestamps_nrd = set.intersection(*common_timestamps_nrd)
    common_timestamps_nrd = np.array(list(common_timestamps_nrd))

    if verbose >= 1:
        print(
            "\n[Info] common timestamps germany:",
            len(common_timestamps_ger),
            "(",
            get_utc(common_timestamps_ger[0]),
            "-",
            get_utc(common_timestamps_ger[-1]),
            ")",
        )
        print(
            "[Info] common timestamps nordics:",
            len(common_timestamps_nrd),
            "(",
            get_utc(common_timestamps_nrd[0]),
            "-",
            get_utc(common_timestamps_nrd[-1]),
            ")",
        )

    for c in best_regions_ger:
        index_orig = np.arange(len(timestamps[c]))
        idx[c] = index_orig[np.isin(timestamps[c], common_timestamps_ger)]

    for c in best_regions_nrd:
        index_orig = np.arange(len(timestamps[c]))
        idx[c] = index_orig[np.isin(timestamps[c], common_timestamps_nrd)]

    total_production_ger = np.zeros(len(common_timestamps_ger))
    total_estimation_ger = np.zeros((1, len(common_timestamps_ger)))
    total_production_nrd = np.zeros(len(common_timestamps_nrd))
    total_estimation_nrd = np.zeros((1, len(common_timestamps_nrd)))

    for c in best_regions_ger:
        total_production_ger += targets[c][idx[c]]
        total_estimation_ger[0] += estimations[c][0][idx[c]]

    for c in best_regions_nrd:
        total_production_nrd += targets[c][idx[c]]
        total_estimation_nrd[0] += estimations[c][0][idx[c]]

    maape = calculate_maape(total_estimation_ger, total_production_ger)[0]
    mape = calculate_mape(total_estimation_ger, total_production_ger, region="DEU")[0]
    mae = calculate_mae(total_estimation_ger, total_production_ger)[0]
    me = calculate_me(total_estimation_ger, total_production_ger)[0]

    print(
        "\n{:}{:>16}: {:8} {:8} {:8} {:8}{:}".format(
            bcolors.BOLD, "region", "maape", "mape", "mae", "me", bcolors.ENDC
        )
    )
    print(
        "{:>16}: {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
            "germany", maape, mape, mae, me
        )
    )

    maape = calculate_maape(total_estimation_nrd, total_production_nrd)[0]
    mape = calculate_mape(total_estimation_nrd, total_production_nrd, region="NRD")[0]
    mae = calculate_mae(total_estimation_nrd, total_production_nrd)[0]
    me = calculate_me(total_estimation_nrd, total_production_nrd)[0]

    print(
        "{:>16}: {:8.3f} {:8.3f} {:8.3f} {:8.3f}".format(
            "nordics", maape, mape, mae, me
        )
    )
    return metrics


def compare_baselines(db, param, verbose):
    ID = 4
    regions_ger = ["EONon", "EONoff", "Vattenfalloff", "Vattenfallon", "RWE", "ENBW"]
    regions_nrd = [
        "DK1",
        "DK1on",
        "DK1off",
        "DK2on",
        "DK2off",
        "NO2",
        "NO3",
        "NO4",
        "SE1",
        "SE2",
        "SE3",
        "SE4",
        "FIN",
    ]
    regions = regions_ger + regions_nrd

    targets = {}
    timestamps = {}
    idx = {}
    estimations = {}

    # comment selection
    # selected_comments_ger = select_comments(db, param, comment, ID, regions_ger)
    # selected_comments_nrd = select_comments(db, param, comment, ID, regions_nrd)

    for region in regions:
        for tr in ["", ".nt"]:
            c = f"{region}.{ID}.{comment}{tr}"
            param["region"] = region
            param["comment"] = c
            targets[c] = db.get(**param, target=True)
            timestamps[c] = db.get(**param, is_timestamps=True)
            estimations[c] = db.get(**param, target=False)

    # only one target. pick first one
    for region in regions:
        for tr in ["", ".nt"]:
            c = f"{region}.{ID}.{comment}{tr}"
            if len(targets[c].shape) == 2:
                targets[c] = targets[c][0]
                if verbose > 1:
                    print("warning: multiple targets for {}".format(c))
            estimations[c] = np.expand_dims(estimations[c], axis=0)
            if region in regions_ger:
                estimations[c] *= 1  # 1.05
            if region in regions_nrd:
                estimations[c] *= 1  # 1.03

    best_regions_ger = []
    best_regions_nrd = []


def main(args):
    db = TestResultsDatabase(verbose=args.verbose)
    db.connect()

    try:
        subprocess.call(["mkdir", "-p", "out"])
    except:
        print(
            "[Warning] cannot create output directory. Correct permissions? (linux/macos)"
        )

    verbose = args.verbose
    param = strip_args(args)

    if args.baselines:
        eval_baselines_wrapper(db, param, verbose)
    elif args.sn_vs_ord:
        check_4_5_agains_baselines(db, param, verbose)
    elif args.autocorr:
        autocorrelation()
    elif args.misc:
        misc(db, param, verbose)
    elif args.test_ns:
        test_ns(db, param, verbose)
    elif args.region == "DEU":
        eval_deu(db, param, verbose)
    elif args.region == "NRD":
        eval_nrd(db, param, verbose)
    elif args.comment2 != None:
        ttest(db, param, verbose)
    else:
        try:
            caclulate_statistics(db, param, verbose)
        except:
            pass

    db.close()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluation script for the Wind power prediction master thesis. This script is contains a collection of tools used to generate figures and analyze the results of the tests."
    )
    model_group = parser.add_argument_group(title="model")
    data_group = parser.add_argument_group(title="dataset")
    flag_group = parser.add_argument_group(title="flags")
    other_group = parser.add_argument_group(title="other")

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="increase output verbosity"
    )
    parser.add_argument(
        "--runs",
        type=int,
        metavar="N",
        help="""Number of repeated training/testing iterations to do.
                This is usefull when the model accuracy is evaluated as the
                only part changins is the initial weights.""",
    )
    parser.add_argument(
        "--comment", type=str, help="Comment appended to result on storage"
    )

    data_group.add_argument("--region1", type=str, help="region to use (default: None)")
    data_group.add_argument("--region2", type=str, help="region to use (default: None)")
    parser.add_argument(
        "--comment1", type=str, help="Comment appended to result on storage"
    )
    parser.add_argument(
        "--comment2", type=str, help="Comment appended to result on storage"
    )

    model_group.add_argument(
        "--batch-size", type=int, metavar="N", help="batch size (default: 32)",
    )
    model_group.add_argument(
        "--epochs", type=int, help="upper epoch limit (default: 100)"
    )
    # parser.add_argument('--log-interval', type=int, default=100, metavar='N',
    #                    help='report interval (default: 100')
    model_group.add_argument(
        "--lr",
        type=float,
        help="learning rate (default: 0.01 for SDG, 0.0001 for ADAM)",
    )
    model_group.add_argument(
        "--optim",
        type=str,
        choices=["SGD", "ADAM"],
        help="optimizer to use (default: SGD)",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        metavar="N",
        help="""probability of dropout in the last fully connected
                layers. (default: 0.2)""",
    )
    model_group.add_argument("--seed", type=int, help="random seed (default: 0)")
    model_group.add_argument(
        "--ordinal-resolution",
        type=int,
        metavar="N",
        help="""number of ordinal classification classes.
                The classes are equally distributed. (default: 100)""",
    )

    data_group.add_argument("--region", type=str, help="region to use (default: None)")
    data_group.add_argument(
        "--normalize-type",
        type=int,
        choices=[1, 2, 3, 4],
        metavar="N",
        help="""normalization type. see report for details. ignored if
                --no-normalize is used. (default: 1)""",
    )
    data_group.add_argument(
        "--window-size",
        type=int,
        metavar="N",
        help="""number of hours before and after the actual timestamp to include
                in estimation. (default: 2)""",
    )
    data_group.add_argument(
        "--timestamp-start", type=int, metavar="N", help="timestamp of first datapoint",
    )
    data_group.add_argument(
        "--timestamp-end", type=int, metavar="N", help="timestamp of last datapoint",
    )
    data_group.add_argument(
        "--data-length", type=int, metavar="N", help="length of dataset",
    )

    flag_group.add_argument(
        "--no-comment", action="store_true", help="""find samples without a comment."""
    )

    flag_group.add_argument(
        "--hybrid", action="store_true", help="""set hybrid to true. default false."""
    )
    flag_group.add_argument(
        "--include-nt",
        action="store_true",
        help="""set include models trained on non-transformed ratio. Only valid for baselines. default false.""",
    )
    flag_group.add_argument(
        "--figures", action="store_true", help="""store figures. default false."""
    )
    flag_group.add_argument(
        "--baselines",
        action="store_true",
        help="""calculate statistics for baselines. default false.""",
    )
    flag_group.add_argument(
        "--multi-gpu",
        "--no-multi-gpu",
        dest="multi-gpu",
        action=NegateAction,
        nargs=0,
        help="""use multiple GPUs if available.
                The data will be split evenly across the GPUs.""",
    )
    flag_group.add_argument(
        "--cuda",
        "--no-cuda",
        dest="cuda",
        action=NegateAction,
        nargs=0,
        help="toggle CUDA",
    )
    flag_group.add_argument(
        "--shuffle",
        "--no-shuffle",
        dest="shuffle",
        action=NegateAction,
        nargs=0,
        help="""toggle shuffeling of dataset. Remember that the shuffle
                is deterministic given the seed.""",
    )
    flag_group.add_argument(
        "--transform",
        "--no-transform",
        dest="ratio_transform",
        action=NegateAction,
        nargs=0,
        help="toggle ratio transformation. see report for details.",
    )
    flag_group.add_argument(
        "--normalize",
        "--no-normalize",
        dest="normalize",
        action=NegateAction,
        nargs=0,
        help="toggle normalization",
    )

    flag_group.add_argument(
        "--lgbm",
        "--no-lgbm",
        dest="is_lightgbm",
        action=NegateAction,
        nargs=0,
        help="toggle lgbm",
    )
    flag_group.add_argument(
        "--use-seed",
        "--no-use-seed",
        dest="use_seed",
        action=NegateAction,
        nargs=0,
        help="toggle the use of random seed",
    )
    flag_group.add_argument(
        "--ordinal",
        "--no-ordinal",
        dest="ordinal",
        action=NegateAction,
        nargs=0,
        help="toggle ordinal regression and use norman one-node regression as target.",
    )
    flag_group.add_argument(
        "--storage",
        "--no-storage",
        dest="store_results",
        action=NegateAction,
        help="toggle storage of model performance.",
    )
    flag_group.add_argument(
        "--latex", action="store_true", help="""output latex tables"""
    )
    flag_group.add_argument(
        "--test-ns",
        action="store_true",
        help="""test normalization atrategies. ignored if --baselines is set""",
    )
    flag_group.add_argument(
        "--misc", action="store_true", help="""run misc test and plots"""
    )
    flag_group.add_argument(
        "--autocorr", action="store_true", help="""generate autocorrelation plots"""
    )
    flag_group.add_argument(
        "--sn-vs-ord",
        action="store_true",
        help="""compare single node regression and ordinal classification """,
    )

    other_group.add_argument(
        "--rank-metric",
        type=str,
        default="maape",
        choices=["maape", "mape", "mae", "me"],
        help="""Metric used to rank best model on validation set (default: maape)""",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.use_seed:
        np.random.seed(args.seed)

    main(args)
