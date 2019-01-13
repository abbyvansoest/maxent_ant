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
    if not np.array_equal(obs[:len(state) - 2], state[2:]):
        print(obs)
        print(state)
        raise ValueError("state and observation are not equal")
    return state

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], n=10, render=False):
    
    buf = ExperienceBuffer()
    
    average_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    average_p_full_dim = np.zeros(shape=(ant_utils.num_states_full))
    random_initial_state = []

    denom = 0
    
    max_idx = len(policies) - 1

    # average results over n rollouts
    for iteration in range(n):
        
        env.reset()
        
        if len(initial_state) > 0:
            qpos = initial_state[:len(ant_utils.qpos)]
            qvel = initial_state[len(ant_utils.qpos):]
            env.env.set_state(qpos, qvel)

        obs = get_state(env, env.env._get_obs())

        random_T = np.floor(random.random()*T)
        random_initial_state = []
       
        for t in range(T):
            
            action = np.zeros(shape=(1,ant_utils.action_dim))
            
            # average the mu
            # take max sigma 
            if args.max_sigma:
                mu = np.zeros(shape=(1,ant_utils.action_dim))
                sigma = np.zeros(shape=(1,ant_utils.action_dim))
                mean_sigma = np.zeros(shape=(1,ant_utils.action_dim))
                for sac in policies:
                    mu += sac.get_action(obs, deterministic=True)
                    sigma = np.maximum(sigma, sac.get_sigma(obs))
                    mean_sigma += sac.get_sigma(obs)
                mu /= float(len(policies))
                mean_sigma /= float(len(policies))

                action = np.random.normal(loc=mu, scale=sigma)
            else:
                # select random policy uniform distribution
                # take non-deterministic action for that policy
                idx = random.randint(0, max_idx)
                action = policies[idx].get_action(obs, deterministic=False)
            
            obs, reward, done, _ = env.step(action)
            obs = get_state(env, obs)
            average_p[tuple(ant_utils.discretize_state(obs))] += 1
            average_p_full_dim[tuple(ant_utils.discretize_state_full(obs))] += 1
            
            buf.store(obs)
            
            denom += 1
            
            if t == random_T:
                random_initial_state = obs

            if render:
                env.render()
            if done:
                env.reset()

    env.close()

    average_p /= float(denom)
    average_p_full_dim /= float(denom)

    buff_p = buf.get_discrete_distribution()
    buff_p_test_full = buf.get_discrete_distribution_full()

    return buff_p, entropy(buff_p_test_full.ravel()), random_initial_state, buff_p_test_full, buf.normalization_factors

def entropy(pt):
    return scipy.stats.entropy(pt)

def grad_ent(pt):
    if args.grad_ent:
        grad_p = -np.log(pt)
        grad_p[grad_p > 100] = 1000
        return grad_p

    eps = 1/np.sqrt(ant_utils.total_state_space)
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
    seed = init_state(env)
    
    # x=0, y=0 is starting seed state for all dimensions.
    v_x = ant_utils.discretize_value(0, ant_utils.state_bins[0])
    v_y = ant_utils.discretize_value(0, ant_utils.state_bins[1])
    reward_fn[tuple((v_x, v_y))] = 1
    
#     # try random execution
#     sac = AntSoftActorCritic(lambda : gym.make(args.env), xid=0,
#             seed=args.seed)
#     seed, _ = sac.test_agent_random(T)
#     reward_fn = grad_ent(seed)
    
    print(reward_fn.shape)
    print(tuple(ant_utils.discretize_state(seed)))
    print(reward_fn[tuple(ant_utils.discretize_state(seed))])

    running_avg_p_full_dim = np.zeros(shape=(tuple(ant_utils.num_states_full)))
    running_avg_ent = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_baseline_full_dim = np.zeros(shape=(tuple(ant_utils.num_states_full)))
    running_avg_ent_baseline = 0

    baseline_ps_full_dim = []
    
    entropies = []
    ps = []

    average_entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps_full_dim = []

    running_avg_entropies_baseline = []
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
            logger_kwargs=logger_kwargs, normalization_factors=normalization_factors, learn_reduced=args.learn_reduced)
        # TODO: start learning from initial state to add gradient?
        sac.soft_actor_critic(epochs=args.episodes, initial_state=initial_state) 
        policies.append(sac) # TODO: save to file

        # CHANGED DETERMNISTIC HERE
        # CHANGED ORDER OF BASELINE COLLECTION AND NORMALIZATION PARAMS
        _, p_full_dim = sac.test_agent(T, deterministic=True, store_log=False, n=args.n) # TODO: initial state seed?

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        average_p, round_avg_ent, initial_state, average_p_full_dim, normalization_factors = \
            execute_average_policy(env, policies, T, n=args.n, render=False)
        
        p_baseline, p_baseline_full = sac.test_agent_random(T, normalization_factors=normalization_factors, n=args.n)
        round_entropy_baseline = entropy(p_baseline_full.ravel())
        
        print(average_p_full_dim)
        print(p_baseline_full)

        average_ps.append(average_p)
        average_entropies.append(round_avg_ent) 

        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p_full_dim = running_avg_p_full_dim * (i)/float(i+1) + average_p_full_dim/float(i+1)
        
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps_full_dim.append(running_avg_p_full_dim) 

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_p_baseline_full_dim = running_avg_p_baseline_full_dim * (i)/float(i+1) + p_baseline_full/float(i+1)
        
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline_full_dim.append(running_avg_p_baseline_full_dim) 
        
        # update reward function
        reward_fn = grad_ent(average_p)
        
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy", "running_avg_ent", "full_dim"]
        col2 = [round_entropy_baseline, running_avg_ent_baseline, entropy(running_avg_p_baseline_full_dim.ravel())]
        col3 = [round_avg_ent, running_avg_ent, entropy(running_avg_p_full_dim.ravel())]
        table = tabulate(np.transpose([col1, col2, col3]), col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        print(table)
        
        # NOTE: the full_dim can only be over 2 dimensions (set start/stop in ant_utils)
        plotting.heatmap(running_avg_p_full_dim, average_p_full_dim, i)
        plotting.heatmap1(running_avg_p_baseline_full_dim, i)
        plotting.heatmap1(p_full_dim, i, directory='p')

    indexes = [0, 5, 10, 20]
    plotting.heatmap4(running_avg_ps_full_dim, running_avg_ps_baseline_full_dim, indexes)
    plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
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
    overall_avg_ent = entropy([1])

    print('*************')
    print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


