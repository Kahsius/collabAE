#!/home/denis/python/collabAE/bin/python

import pdb
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from results_wdbc import get_results
matplotlib.rcParams.update({'font.size': 15})


def plot(g, gg, title, filename, name, g3=0):
    n_groups = len(g)

    fig, ax = plt.subplots()
    liminf = min(min(g), min(gg), 0) * 1.1
    limsup = max(max(g), max(gg))
    limsup = limsup * 1.3 if type(g3) == type(0) else limsup * 1.5
    plt.gca().set_ylim([liminf, limsup])

    index = np.arange(n_groups)
    bar_width = 0.35 if type(g3) == type(0) else 0.23

    opacity = 1
    error_config = {'ecolor': '0.3'}

    if type(g3) != type(0):
        rects3 = ax.bar(
            index,
            g3,
            bar_width,
            alpha=opacity,
            color='#222222',
            error_kw=error_config,
            label='Original')

    rects1 = ax.bar(
        index + bar_width,
        g,
        bar_width,
        alpha=opacity,
        color='#666666',
        error_kw=error_config,
        label='Without MWM')

    rects2 = ax.bar(
        index + bar_width * 2,
        gg,
        bar_width,
        alpha=opacity,
        color='#BBBBBB',
        error_kw=error_config,
        label='With MWM')

    name2 = name.upper() if name != 'mfeat' else 'MFDD'
    name2 = 'Madelon' if name == 'madelon' else name2
    name2 = 'Cube' if name == 'cube' else name2
    ax.set_xlabel('Views')
    ax.set_ylabel(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(range(n_groups)))
    ax.legend()

    fig.tight_layout()
    fig.savefig("img/" + name + "/" + filename + ".pdf")
    print("img/" + name + "/" + filename + ".pdf")


def generate_all_graphs(name):
    # WDBC
    c, c2, a, a2, l, l2, fm, fm2, fr, fr2, fc, fc2 = get_results(
        "data/" + name + "/results_sans_weights/")
    cc, cc2, aa, aa2, ll, ll2, ffm, ffm2, ffr, ffr2, ffc, ffc2 = get_results(
        "data/" + name + "/results_standard/")
    m = np.mean([np.asarray(cc),np.asarray(c)], axis=0)
    plot(fm, ffm, "Mean Squared Error", "mse_" + name, name)
    plot(fr, ffr, "Mean Relative Difference", "mrd_" + name, name)
    plot(fc, ffc, "Classification Score", "cs_" + name, name, g3=m)
    plot(
        np.asarray(fc) - m,
        np.asarray(ffc) - m,
        "Classification Difference",
        "cd_" + name,
        name,
    )
    print("Max diff : " + str(np.max(np.abs(np.asarray(ffc)-m))))
    print("Mean diff : " + str(np.mean(np.abs(np.asarray(ffc)-m))))
    print("Mean original " + str(np.mean(m)))

for i in range(len(sys.argv)-1):
    generate_all_graphs(sys.argv[i+1])
