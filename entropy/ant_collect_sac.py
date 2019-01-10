# Collect entropy-based reward policies.

# python ant_collect_sac.py --env="Ant-v2" --T=1000 --episodes=100 --epochs=10 

import os
import time
from datetime import datetime

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm
from tabulate import tabulate
import sys 

import gym
from gym.spaces import prng
import tensorflow as tf

import utils
import ant_utils
import plotting
from ant_soft_actor_critic import AntSoftActorCritic
from experience_buffer import ExperienceBuffer

import torch
from torch.distributions import Normal
import random

args = utils.get_args()

from spinup.utils.run_utils import setup_logger_kwargs

def get_state(env, obs):
    state = env.env.state_vector()
    return state

def print_dimensions(full_dim):
    print(full_dim.shape)
    print(full_dim)
    
    for i in range(ant_utils.stop - ant_utils.start):
        dim = np.sum(full_dim, axis=i)
        dim_entropy = scipy.stats.entropy(dim.flatten())
        print(dim)
        print("dim_entropy[%d] = %.4f" % (i, dim_entropy))

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], n=10, render=False):
    
    buffer = ExperienceBuffer()
    
    average_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    average_p_full_dim = np.zeros(shape=(ant_utils.num_states_full))
    avg_entropy = 0
    random_initial_state = []

    denom = 0
    
    max_idx = len(policies) - 1

    # average results over n rollouts
    for iteration in range(n):
        
        env.reset()
        
        if len(initial_state) > 0:
            qpos = initial_state[:15]
            qvel = initial_state[15:]
            env.env.set_state(qpos, qvel)

        obs = env.env._get_obs()
        p = np.zeros(shape=(tuple(ant_utils.num_states)))
        p_full_dim = np.zeros(shape=(ant_utils.num_states_full))

        random_T = np.floor(random.random()*T)
        random_initial_state = []
       
        inner_denom = 0
        for t in range(T):
            
            action = np.zeros(shape=(1,ant_utils.action_dim))
            
            # average the mu
            # take max sigma 
            if args.max_sigma:
                mu = np.zeros(shape=(1,ant_utils.action_dim))
                sigma = np.zeros(shape=(1,ant_utils.action_dim))
                for sac in policies:
                    mu += sac.get_action(obs, deterministic=True)
                    sigma = np.maximum(sigma, sac.get_sigma(obs))
                mu /= len(policies)

                action = np.random.normal(loc=mu, scale=sigma)

                # dist = tfd.Normal(loc=mu, scale=sigma)
                # sess = tf.Session()
                # with sess.as_default():
                #     action = dist.sample([1]).eval().reshape(8)
            else:
                # select random policy uniform distribution
                # take non-deterministic action for that policy
                idx = random.randint(0, max_idx)
                action = policies[idx].get_action(obs, deterministic=False)
            
            obs, reward, done, _ = env.step(action)
            p[tuple(ant_utils.discretize_state(get_state(env, obs)))] += 1
            p_full_dim[tuple(ant_utils.discretize_state_full(get_state(env, obs)))] += 1
            
#             print(obs)
#             print(tuple(ant_utils.discretize_state(get_state(env, obs))))
            
            buffer.store(get_state(env, obs))
            
            denom += 1
            inner_denom += 1
            
            if t == random_T:
                random_initial_state = get_state(env, obs)

            if render:
                env.render()
            if done:
                env.reset()

        average_p += p
        average_p_full_dim += p_full_dim
        avg_entropy += scipy.stats.entropy(average_p.flatten())

    env.close()

    average_p /= float(denom)
    average_p_full_dim /= float(denom)
    avg_entropy /= float(n) # running average of the entropy 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())
    
    buff_p = buffer.get_discrete_distribution()
    buff_p_test_full = buffer.get_discrete_distribution_full()

#     return average_p, avg_entropy, random_initial_state, average_p_full_dim
    return buff_p, scipy.stats.entropy(buff_p.flatten()), random_initial_state, buff_p_test_full, buffer.normalization_factors

def grad_ent(pt):
    eps = .001
    return 1/(pt + eps)

def init_state(env):
    env.env.set_state(ant_utils.qpos, ant_utils.qvel)
    state = env.env.state_vector()
    return state

# Main loop of maximum entropy program. WORKING HERE
# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
# Main loop of maximum entropy program. Iteratively collect 
# and learn T policies using policy gradients and a reward function 
# based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR=''):

    reward_fn = np.zeros(shape=(tuple(ant_utils.num_states)))
    print(reward_fn.shape)

    # set initial state to base state.
    seed = init_state(env)
    print("seed = " + str(seed))
    print(tuple(ant_utils.discretize_state(seed)))
    reward_fn[tuple(ant_utils.discretize_state(seed))] = 10

    running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_full_dim = np.zeros(shape=(tuple(ant_utils.num_states_full)))
    running_avg_ent = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_baseline_full_dim = np.zeros(shape=(tuple(ant_utils.num_states_full)))
    running_avg_ent_baseline = 0

    baseline_entropies = []
    baseline_ps = []
    baseline_ps_full_dim = []
    
    entropies = []
    ps = []

    average_entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps = []
    running_avg_ps_full_dim = []

    running_avg_entropies_baseline = []
    running_avg_ps_baseline = []
    running_avg_ps_baseline_full_dim = []

    policies = []
    normalization_factors = []
    initial_state = init_state(env)

    for i in range(epochs):

        print("*** EPOCH: " + str(i))

        direct = os.getcwd()+ '/data'
        logger_kwargs = setup_logger_kwargs(args.exp_name+ "/model" + str(i), args.seed, data_dir=direct)

        # Learn policy that maximizes current reward function.
        sac = AntSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
            seed=args.seed, gamma=args.gamma, 
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            logger_kwargs=logger_kwargs, normalization_factors=normalization_factors)
        # TODO: start learning from initial state to add gradient?
        sac.soft_actor_critic(epochs=args.episodes, initial_state=initial_state) 
        policies.append(sac) # TODO: save to file

        p, _ = sac.test_agent(T, deterministic=False, store_log=False, n=10) # TODO: initial state seed?

        round_entropy = scipy.stats.entropy(p.flatten())
        entropies.append(round_entropy)
        ps.append(p)

        p_baseline, p_baseline_full = sac.test_agent_random(T, n=10)
        round_entropy_baseline = scipy.stats.entropy(p_baseline.flatten())

        baseline_entropies.append(round_entropy_baseline)
        baseline_ps.append(p_baseline)

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        average_p, round_avg_ent, initial_state, average_p_full_dim, normalization_factors = \
            execute_average_policy(env, policies, T, render=False)
        
        print(average_p_full_dim)
        print(p_baseline_full)

        average_ps.append(average_p)
        average_entropies.append(round_avg_ent) 

        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p = running_avg_p * (i)/float(i+1) + average_p/float(i+1)
        running_avg_p_full_dim = running_avg_p_full_dim * (i)/float(i+1) + average_p_full_dim/float(i+1)
        
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p) 
        running_avg_ps_full_dim.append(running_avg_p_full_dim) 

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_p_baseline = running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)
        running_avg_p_baseline_full_dim = running_avg_p_baseline_full_dim * (i)/float(i+1) + p_baseline_full/float(i+1)
        
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline.append(running_avg_p_baseline) 
        running_avg_ps_baseline_full_dim.append(running_avg_p_baseline_full_dim) 
        
        # update reward function
        reward_fn = grad_ent(average_p)
        
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy", "round_mixed_ent", "running_avg_ent", "entropy_of_running_p"]
        ent = scipy.stats.entropy(running_avg_p_baseline.flatten())
        col2 = [round_entropy_baseline, "", running_avg_ent_baseline, ent]
        col3 = [round_entropy, round_avg_ent, running_avg_ent, scipy.stats.entropy(running_avg_p.flatten())]
        table = tabulate(np.transpose([col1, col2, col3]), col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        print(table)
        
        # NOTE: the full_dim can only be over 2 dimensions (set start/stop in ant_utils)
        plotting.heatmap(running_avg_p_full_dim, average_p_full_dim, i)
        plotting.heatmap1(running_avg_p_baseline_full_dim, i)

#     indexes = [0, 2, 5, 10]
    indexes = []
    print('which indexes?')
    for i in range(4):
        idx = input("index :")
        indexes.append(int(idx))
    plotting.heatmap4(running_avg_ps_full_dim, running_avg_ps_baseline_full_dim, indexes)
    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100, linewidth=150, precision=4)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space
    
    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    plotting.FIG_DIR = 'figs/' + args.env + '/'
    plotting.model_time = args.exp_name
    if not os.path.exists(plotting.FIG_DIR+plotting.model_time):
        os.makedirs(plotting.FIG_DIR+plotting.model_time)

    policies = collect_entropy_policies(env, args.epochs, args.T)

    # average_p = exploration_policy.execute(args.T, render=True)
    overall_avg_ent = scipy.stats.entropy([1])

    print('*************')
    print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


