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

def running_average_entropy(running_avg_entropies, running_avg_entropies_baseline, ext=''):
    fname = get_next_file(FIG_DIR, model_time, "running_avg"+ext, ".png")
    plt.figure()
    plt.plot(np.arange(len(running_avg_entropies)), running_avg_entropies)
    plt.plot(np.arange(len(running_avg_entropies_baseline)), running_avg_entropies_baseline)
    plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Running average entropy of mixed policy")
    plt.title("Policy Entropy over Time")
    plt.savefig(fname)
    plt.close()

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
    plt.close()

def heatmap1(avg_p, i, directory='baseline'):
    # Create running average heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(avg_p))
    if min_value == 0:
        plt.imshow(avg_p, interpolation='spline16', cmap='Oranges')
    else:
        plt.imshow(np.ma.log(avg_p).filled(min_value), interpolation='spline16', cmap='Oranges')

    plt.xticks([], [])
    plt.yticks([], [])
            
    if (args.env == "Ant-v2"):
        plt.xlabel(ant_utils.dim_dict[ant_utils.start])
        plt.ylabel(ant_utils.dim_dict[ant_utils.start+1])
        
    baseline_heatmap_dir = FIG_DIR + model_time + directory + '/'
    if not os.path.exists(baseline_heatmap_dir):
        os.makedirs(baseline_heatmap_dir)
    fname = baseline_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)
    plt.close()

def heatmap(running_avg_p, avg_p, i):
    # Create running average heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(running_avg_p))
    if min_value == 0:
        plt.imshow(running_avg_p, interpolation='spline16', cmap='Blues')
    else:
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
    running_avg_heatmap_dir = FIG_DIR + model_time + 'running_avg/'
    if not os.path.exists(running_avg_heatmap_dir):
        os.makedirs(running_avg_heatmap_dir)
    fname = running_avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)

    # Create episode heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(avg_p))
    if min_value == 0:
        plt.imshow(avg_p, interpolation='spline16', cmap='Blues')
    else:
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
    avg_heatmap_dir = FIG_DIR + model_time + 'avg/'
    if not os.path.exists(avg_heatmap_dir):
        os.makedirs(avg_heatmap_dir)
    fname = avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)
    plt.close()


def heatmap4(running_avg_ps, running_avg_ps_baseline, indexes=[0,1,2,3], ext=''):
    
    plt.figure()
    row1 = [plt.subplot(241), plt.subplot(242), plt.subplot(243), plt.subplot(244)]
    row2 = [plt.subplot(245), plt.subplot(246), plt.subplot(247), plt.subplot(248)]
    
    idx = 0
    for epoch, ax in zip(indexes,row1):
        print(epoch)
        print(idx)
        min_value = np.min(np.ma.log(running_avg_ps[idx]))
        
        if min_value == 0:
            ax.imshow(running_avg_ps[idx], interpolation='spline16', cmap='Blues')
        else:
            ax.imshow(np.ma.log(running_avg_ps[idx]).filled(min_value), interpolation='spline16', cmap='Blues')
        ax.set_title("Epoch %d" % epoch)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        idx += 1
    
    idx = 0
    for epoch, ax in zip(indexes,row2):
        min_value = np.min(np.ma.log(running_avg_ps_baseline[idx]))
        if min_value == 0:
            ax.imshow(running_avg_ps_baseline[idx], interpolation='spline16', cmap='Oranges')
        else:
            ax.imshow(np.ma.log(running_avg_ps_baseline[idx]).filled(min_value), interpolation='spline16', cmap='Oranges')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        idx += 1

    plt.tight_layout()
    fname = get_next_file(FIG_DIR, model_time, ext+"_heatmaps", ".png")
    plt.savefig(fname)
    plt.close()
    
def reward_vs_t(reward_at_t, epoch):
    
    plt.figure()
    plt.plot(np.arange(len(reward_at_t)), reward_at_t)
#     plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Objetive function")
#     plt.title("Objective function return for mixed policy")
    
    t_dir = FIG_DIR + model_time + 't_rewards/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    fname = t_dir + "epoch_%02d.png" % (epoch)
    plt.savefig(fname)
    plt.close()

def percent_state_space_reached(pcts, pcts_baseline, ext=''):
    plt.figure()
    plt.plot(np.arange(len(pcts)), pcts)
    plt.plot(np.arange(len(pcts_baseline)), pcts_baseline)
    plt.xlabel("t")
    plt.ylabel("Fraction of state space reached")
    plt.legend(["MaxEnt", "Random"])
    fname = FIG_DIR + model_time + 'pct_visited' + ext + '.png'
    plt.savefig(fname)
    plt.close()
    
def states_visited_over_time(states_visited, states_visited_baseline, epoch, ext=''):
    plt.figure()
    plt.plot(np.arange(len(states_visited)), states_visited)
    plt.plot(np.arange(len(states_visited_baseline)), states_visited_baseline)
    plt.legend(["Entropy", "Random"])
    plt.xlabel("t")
    plt.ylabel("Number of unique states visited")
    states_dir = FIG_DIR + model_time + 'states_visited' + ext + '/'
    if not os.path.exists(states_dir):
        os.makedirs(states_dir)
    fname = states_dir + "epoch_%02d.png" % (epoch)
    plt.savefig(fname)
    plt.close()