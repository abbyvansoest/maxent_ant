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

from ant_soft_actor_critic import AntSoftActorCritic

import utils
import ant_utils
import plotting

import torch
from torch.distributions import Normal
import random

args = utils.get_args()

from spinup.utils.run_utils import setup_logger_kwargs

# f = open(args.models_dir,'w'); sys.stdout = f

def get_state(env, obs):
    state = env.env.state_vector()
    return state

def print_dimensions(full_dim):
    print(full_dim.shape)
    print(full_dim[0].shape)    
    
    x = np.sum(full_dim, axis=0)
    y = np.sum(full_dim, axis=1)
    z = np.sum(full_dim, axis=2)
    
    x_entropy = scipy.stats.entropy(x.flatten())
    print(x)
    print("x_entropy = %.4f" % x_entropy)
    
    y_entropy = scipy.stats.entropy(y.flatten())
    print(y)
    print("y_entropy = %.4f" % y_entropy)
    
    print(z)
    z_entropy = scipy.stats.entropy(z.flatten())
    print("z_entropy = %.4f" % z_entropy)
    
# get weight averaged model from the provided (identically formatted) tf models. 
# can provide an arbitrary number of models.
def run_average_model(models, T):
    
    average_sac = AntSoftActorCritic(lambda : gym.make(args.env), xid=0,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))
    
    average_sess = average_sac.sess
    
    values = []
    all_vars = tf.trainable_variables()
    for sac in models:
        values.append(sac.sess.run(all_vars))

    average_assign = []
    for i in range(len(all_vars)):
        vals = [val[i] for val in values]
        average_value = np.average(vals)
        average_assign.append(all_vars[i], average_value)

    averaged_values = average_sess.run(average_assign)
    
    average_sac.sess = average_sess
    _, p_averaged_full_dim = average_sac.test_agent(T, store_log=False, deterministic=False)
    
    print_dimensions(p_averaged_full_dim)
    

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], n=10, render=False):
    
    average_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    avg_entropy = 0
    random_initial_state = []

    denom = 0
    
    # collect the full distribution over all states here.
    # later, slice it into each individual distribution and print.
    average_p_full_dim = np.zeros(shape=(ant_utils.num_states_full))

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

        # TODO: examine the behavior of the individual policies
       
        inner_denom = 0
        for t in range(T):

            action = np.zeros(shape=(1,ant_utils.action_dim))
            for i, sac in enumerate(policies):
                a = sac.get_action(obs, deterministic=False)
                # a_det = sac.get_action(obs, deterministic=True) # question: is this returning the same value?
                # print("non-det: %s \t det: %s" % (str(a), str(a_det)))
                action += a
                #print("policy %d: %s" % (i, str(a_det)))
            action /= len(policies)
            #print("final action: %s" % str(action))
            
            obs, reward, done, _ = env.step(action)
            p[tuple(ant_utils.discretize_state(get_state(env, obs)))] += 1
            # print(tuple(ant_utils.discretize_state_full(get_state(env, obs))))
            p_full_dim[tuple(ant_utils.discretize_state_full(get_state(env, obs)))] += 1
            
            denom += 1
            inner_denom += 1
            
            if t == random_T:
                random_initial_state = get_state(env, obs)

            if render:
                env.render()
            if done:
                env.reset()

#         print("trial %d: %s" % (iteration, str(p_full_dim / float(inner_denom))))
        # print("~~~~~~~~~~~~~~~~~~~~")
        average_p += p
        average_p_full_dim += p_full_dim
        avg_entropy += scipy.stats.entropy(average_p.flatten())

    env.close()

    average_p /= float(denom)
    average_p_full_dim /= float(denom)
    avg_entropy /= float(n) # running average of the entropy 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # Get representations of each of the 29 dimensions and save.
    print_dimensions(average_p_full_dim)

    return average_p, avg_entropy, random_initial_state

def grad_ent(pt):
    eps = .001
    return 1/(pt + eps)

def init_state(env):
    env.env.set_state(ant_utils.qpos, ant_utils.qvel)
    state = env.env.state_vector()
    # state[2] = 0.2 # start from the bottom
    return state

# Main loop of maximum entropy program. WORKING HERE
# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
# Main loop of maximum entropy program. Iteratively collect 
# and learn T policies using policy gradients and a reward function 
# based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR=''):

    reward_fn = np.zeros(shape=(tuple(ant_utils.num_states)))

    # set initial state to base state.
    seed = init_state(env)
    reward_fn[tuple(ant_utils.discretize_state(seed))] = 10
    print("seed = " + str(seed))
    print(tuple(ant_utils.discretize_state(seed)))

    print(reward_fn.shape)

    running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_ent = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_ent_baseline = 0

    baseline_entropies = []
    baseline_ps = []
    entropies = []
    ps = []

    average_entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps = []

    running_avg_entropies_baseline = []
    running_avg_ps_baseline = []

    policies = []
    initial_state = init_state(env)

    for i in range(epochs):

        print("*** EPOCH: " + str(i))

        logger_kwargs = setup_logger_kwargs(args.exp_name+ "/model" + str(i), args.seed, data_dir='/home/abby/entropy/data')

        # Learn policy that maximizes current reward function.
        sac = AntSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
            seed=args.seed, gamma=args.gamma, 
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            logger_kwargs=logger_kwargs)
        sac.soft_actor_critic(epochs=args.episodes, initial_state=initial_state) # TODO: start learning from initial state to add gradient
        policies.append(sac) # TODO: save to file

        p, _ = sac.test_agent(T, deterministic=False, n=10) # TODO: initial state seed?

        round_entropy = scipy.stats.entropy(p.flatten())
        entropies.append(round_entropy)
        ps.append(p)

        p_baseline = sac.test_agent_random(T, n=10)
        round_entropy_baseline = scipy.stats.entropy(p_baseline.flatten())

        baseline_entropies.append(round_entropy_baseline)
        baseline_ps.append(p_baseline)

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        average_p, round_avg_ent, initial_state = \
            execute_average_policy(env, policies, T, render=False)

        average_ps.append(average_p)
        average_entropies.append(round_avg_ent) 

        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p = running_avg_p * (i)/float(i+1) + average_p/float(i+1)
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)     

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_p_baseline = running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline.append(running_avg_p_baseline) 
        
        # update reward function
        old_reward_fn = reward_fn
        reward_fn = grad_ent(running_avg_p) # grad_ent(average_p)

        # print("---------------------")
        
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy", "round_mixed_ent", "running_avg_ent", "entropy_of_running_p"]
        col2 = [round_entropy_baseline, "", running_avg_ent_baseline, scipy.stats.entropy(running_avg_p_baseline.flatten())]
        col3 = [round_entropy, round_avg_ent, running_avg_ent, scipy.stats.entropy(running_avg_p.flatten())]
        table = tabulate(np.transpose([col1, col2, col3]), col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        print(table)

        print("----------- Testing Model Averaging -----------")
        
        # average models?
        run_average_model(policies, T) # ADDED!!!!!

    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100, linewidth=150, precision=4)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    policies = collect_entropy_policies(env, args.epochs, args.T)

    # average_p = exploration_policy.execute(args.T, render=True)
    overall_avg_ent = scipy.stats.entropy([1])

    print('*************')
    print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


