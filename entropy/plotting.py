import os

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg') # matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

import utils
import ant_utils
args = utils.get_args()

# By default, the plotter saves figures to the directory where it's executed.
FIG_DIR = ''
model_time = ''

def get_next_file(directory, model_time, ext, dot=".png"):
    i = 0
    fname = directory + model_time + ext
    while os.path.isfile(fname):
        fname = directory + model_time + ext + str(i) + dot
        i += 1
    return fname

def running_average_entropy(running_avg_entropies, running_avg_entropies_baseline):
    fname = get_next_file(FIG_DIR, model_time, "running_avg", ".png")
    plt.figure()
    plt.plot(np.arange(len(running_avg_entropies)), running_avg_entropies)
    plt.plot(np.arange(len(running_avg_entropies_baseline)), running_avg_entropies_baseline)
    plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Running average entropy of cumulative policy")
    plt.title("Policy Entropy over Time")
    plt.savefig(fname)

def running_average_entropy_window(window_running_avg_ents, window_running_avg_ents_baseline, window):
    fname = get_next_file(FIG_DIR, model_time, "running_avg_window", ".png")
    plt.figure()
    plt.plot(np.arange(len(window_running_avg_ents)), window_running_avg_ents)
    plt.plot(np.arange(len(window_running_avg_ents_baseline)), window_running_avg_ents_baseline)
    plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Running avg entropy")
    plt.title("Policy entropy over time, window = %d" % window)
    plt.savefig(fname)

def heatmap1(avg_p, i, directory='baseline'):
    # Create running average heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(avg_p))
    plt.imshow(np.ma.log(avg_p).filled(min_value), interpolation='spline16', cmap='Oranges')

    plt.xticks([], [])
    plt.yticks([], [])
            
    if (args.env == "Ant-v2"):
        plt.xlabel(ant_utils.dim_dict[ant_utils.start])
        plt.ylabel(ant_utils.dim_dict[ant_utils.start+1])
        
    baseline_heatmap_dir = FIG_DIR + model_time + '/' + directory + '/'
    if not os.path.exists(baseline_heatmap_dir):
        os.makedirs(baseline_heatmap_dir)
    fname = baseline_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)

def heatmap(running_avg_p, avg_p, i):
    # Create running average heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(running_avg_p))
    #print(running_avg_p)
    plt.imshow(np.ma.log(running_avg_p).filled(min_value), interpolation='spline16', cmap='Blues')

    plt.xticks([], [])
    plt.yticks([], [])
    
    plt.xlabel("v")
    if (args.env == "Ant-v2"):
        plt.xlabel(ant_utils.dim_dict[ant_utils.start])
        
    if (args.env == "MountainCarContinuous-v0"):
        plt.ylabel("x")
    elif (args.env == "Pendulum-v0"):
        plt.ylabel(r"$\Theta$")
    elif (args.env == "Ant-v2"):
        plt.ylabel(ant_utils.dim_dict[ant_utils.start+1])
        
    # plt.title("Policy distribution at step %d" % i)
    running_avg_heatmap_dir = FIG_DIR + model_time + '/' + 'running_avg' + '/'
    if not os.path.exists(running_avg_heatmap_dir):
        os.makedirs(running_avg_heatmap_dir)
    fname = running_avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)

    # Create episode heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(avg_p))
    #print(avg_p)
    plt.imshow(np.ma.log(avg_p).filled(min_value), interpolation='spline16', cmap='Blues')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("v")
    if (args.env == "Ant-v2"):
        plt.xlabel(ant_utils.dim_dict[ant_utils.start])
        
    if (args.env == "MountainCarContinuous-v0"):
        plt.ylabel("x")
    elif (args.env == "Pendulum-v0"):
        plt.ylabel(r"$\Theta$")
    elif (args.env == "Ant-v2"):
        plt.ylabel(ant_utils.dim_dict[ant_utils.start+1])

    # plt.title("Policy distribution at step %d" % i)
    avg_heatmap_dir = FIG_DIR + model_time + '/' + 'avg' + '/'
    if not os.path.exists(avg_heatmap_dir):
        os.makedirs(avg_heatmap_dir)
    fname = avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)


def heatmap4(running_avg_ps, running_avg_ps_baseline, indexes=[0,1,2,3]):
    plt.figure()
    row1 = [plt.subplot(241), plt.subplot(242), plt.subplot(243), plt.subplot(244)]
    row2 = [plt.subplot(245), plt.subplot(246), plt.subplot(247), plt.subplot(248)]

    # min_value = np.min(np.ma.log(running_avg_ps))
    # min_value_baseline = np.min(np.ma.log(running_avg_ps_baseline))
    # min_value = np.minimum(min_value, min_value_baseline)

    # TODO: colorbar for the global figure
    for idx, ax in zip(indexes,row1):
        min_value = np.min(np.ma.log(running_avg_ps[idx]))
        ax.imshow(np.ma.log(running_avg_ps[idx]).filled(min_value), interpolation='spline16', cmap='Blues')
        ax.set_title("Epoch %d" % idx)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    
    for idx, ax in zip(indexes,row2):
        min_value = np.min(np.ma.log(running_avg_ps_baseline[idx]))
        ax.imshow(np.ma.log(running_avg_ps_baseline[idx]).filled(min_value), interpolation='spline16', cmap='Oranges')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    plt.tight_layout()
    fname = get_next_file(FIG_DIR, model_time+'/', "time_heatmaps", ".png")
    plt.savefig(fname)
    # plt.colorbar()
    # plt.show()

def difference_heatmap(running_avg_ps, running_avg_ps_baseline):

    fname = get_next_file(FIG_DIR, model_time, "heatmap", ".png")
    entropy_p = running_avg_ps[len(running_avg_ps) - 1]
    random_p = running_avg_ps_baseline[len(running_avg_ps_baseline) - 1]

    plt.figure()
    diff_map = entropy_p - random_p

    # normalize so 0 is grey.
    max_p = max(abs(np.min(diff_map)), np.max(diff_map))
    plt.imshow(diff_map, vmin=-max_p, vmax=max_p, interpolation='spline16', cmap='coolwarm')
    plt.colorbar()
    plt.title(r'$p_{\pi_{entropy}} - p_{\pi_{random}}$')
    plt.savefig(fname)
    # plt.show()