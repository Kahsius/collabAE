import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results_wdbc import get_results
matplotlib.rcParams.update({'font.size': 15})

def plot(g, gg, title, filename) :
    n_groups = len(g)

    name = "WDBC" if filename.find("wdbc") != -1 else "MFDD"
    fig, ax = plt.subplots()
    if title == "CS" : plt.gca().set_ylim([0,1.3])

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, g, bar_width,
                    alpha=opacity, color='#666666', error_kw=error_config,
                    label='Without MWM')

    rects2 = ax.bar(index + bar_width, gg, bar_width,
                    alpha=opacity, color='#BBBBBB', error_kw=error_config,
                    label='With MWM')

    ax.set_xlabel('Views of ' + name)
    ax.set_ylabel(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(range(n_groups)))
    ax.legend()

    fig.tight_layout()
    fig.savefig("img/"+filename+".pdf")
    print("img/"+filename+".pdf")

# WDBC
c, c2, a, a2, l, l2, fm, fm2, fr, fr2, fc, fc2 = get_results("data/wdbc/results_sans_weights/")
cc, cc2, aa, aa2, ll, ll2, ffm, ffm2, ffr, ffr2, ffc, ffc2 = get_results("data/wdbc/results_standard/")
plot(fm, ffm, "MSE", "mse_wdbc")
plot(fr, ffr, "MRD", "mrd_wdbc")
plot(fc, ffc, "CS", "cs_wdbc")
plot(np.asarray(fc)-np.asarray(c), np.asarray(ffc)-np.asarray(cc), "CD", "cd_wdbc")

# MFEAT
c, c2, a, a2, l, l2, fm, fm2, fr, fr2, fc, fc2 = get_results("data/mfeat/results_sans_weights/")
cc, cc2, aa, aa2, ll, ll2, ffm, ffm2, ffr, ffr2, ffc, ffc2 = get_results("data/mfeat/results_standard/")
plot(fm, ffm, "MSE", "mse_mfeat")
plot(fr, ffr, "MRD", "mrd_mfeat")
plot(fc, ffc, "CS", "cs_mfeat")
plot(np.asarray(fc)-np.asarray(c), np.asarray(ffc)-np.asarray(cc), "CD", "cd_mfeat")
