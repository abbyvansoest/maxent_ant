# Collect entropy-based reward policies.

# python ant_collect_sac.py --env="Ant-v2" --T=1000 --episodes=100 --epochs=10 

import os
import time
from datetime import datetime

import random
import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm
from scipy.stats import entropy
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
def execute_average_policy(env, policies, T, reward_fn=[], norm=[], initial_state=[], n=10, render=False, epoch=0):
    
#     buf = ExperienceBuffer()
    p = np.zeros(shape=(tuple(ant_utils.num_states)))
    p_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d)))

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
        
#         rewards = []
       
        for t in range(T):
            
            action = np.zeros(shape=(1,ant_utils.action_dim))
            
            # average the mu
            # take max sigma 
#             if max_idx == 0 :
#                 action = policies[0].get_action(obs, deterministic=True)
#             el
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
                action = policies[idx].get_action(obs, deterministic=args.deterministic)
            
            obs, _, done, _ = env.step(action)
            obs = get_state(env, obs)
#             reward = reward_fn[tuple(ant_utils.discretize_state(obs, norm))]
#             rewards.append(reward)

#             buf.store(obs)
            p[tuple(ant_utils.discretize_state(obs, norm))] += 1
            p_xy[tuple(ant_utils.discretize_state_2d(obs, norm))] += 1
            denom += 1
            
            if t == random_T:
                random_initial_state = obs

            if render:
                env.render()
            if done:
                # reset to most recent observation
                env.reset()
#                 qpos = obs[:len(ant_utils.qpos)]
#                 qvel = obs[len(ant_utils.qpos):]
#                 env.env.set_state(qpos, qvel)
                
#                 plotting.reward_vs_t(rewards, epoch, iteration)

    env.close()

#     buff_p = buf.get_discrete_distribution()
#     buff_p_test_xy = buf.get_discrete_distribution_2d()
    p /= float(denom)
    p_xy /= float(denom)

#     return buff_p, random_initial_state, buff_p_test_xy, buf.normalization_factors
    return p, p_xy, random_initial_state

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
    
    direct = os.getcwd()+ '/data/'
    experiment_directory = direct + args.exp_name
    print(experiment_directory)
    indexes = [0, 5, 10, 20]
#     indexes=[0,1,2,3]

    running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d)))
    running_avg_ent = 0
    running_avg_ent_xy = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_p_baseline_xy = np.zeros(shape=(tuple(ant_utils.num_states_2d)))
    running_avg_ent_baseline = 0
    running_avg_ent_baseline_xy = 0

    pct_visited = []
    pct_visited_baseline = []
    pct_visited_xy = []
    pct_visited_xy_baseline = []

    running_avg_entropies = []
    running_avg_ps_xy = []

    running_avg_entropies_baseline = []
    running_avg_ps_baseline_xy = []

    policies = []
    initial_state = init_state(env)
    
    prebuf = ExperienceBuffer()
    env.reset()
    for t in range(10000):     
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        prebuf.store(get_state(env, obs))
        
        if done:
            env.reset()
            done = False
            
    prebuf.normalize()
    normalization_factors = prebuf.normalization_factors
    print(normalization_factors)
    prebuf = None

    reward_fn = np.zeros(shape=(tuple(ant_utils.num_states)))
    seed = init_state(env)
    
    # x=0, y=0 is starting seed state for all dimensions.
    v_x = ant_utils.discretize_value(0, ant_utils.state_bins[0])
    v_y = ant_utils.discretize_value(0, ant_utils.state_bins[1])
    reward_fn[tuple((v_x, v_y))] = 1
    
    print(reward_fn.shape)
    print(tuple(ant_utils.discretize_state(seed, normalization_factors)))
    print(reward_fn[tuple(ant_utils.discretize_state(seed, normalization_factors))])

    for i in range(epochs):
        print("*** ------- EPOCH %d ------- ***" % i)
        
        # clear initial state if applicable.
        if not args.initial_state:
            initial_state = []
        else:
            print(initial_state)
            print(tuple(ant_utils.discretize_state_2d(initial_state, normalization_factors)))
            print(tuple(ant_utils.discretize_state(initial_state, normalization_factors)))
        print("max reward: " + str(np.max(reward_fn)))

        logger_kwargs = setup_logger_kwargs("model" + str(i), data_dir=experiment_directory)

        # Learn policy that maximizes current reward function.
        print("Learning new oracle...")
        if args.seed != -1:
            seed = args.seed
        else:
            seed = random.randint(1, 100000)
        print(type(seed))
        sac = AntSoftActorCritic(lambda : gym.make(args.env), reward_fn=reward_fn, xid=i+1,
            seed=seed, gamma=args.gamma, 
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            logger_kwargs=logger_kwargs, 
            normalization_factors=normalization_factors,
            learn_reduced=args.learn_reduced)
        # TODO: start learning from initial state to add gradient?
        sac.soft_actor_critic(epochs=args.episodes, 
                              initial_state=initial_state, 
                              start_steps=args.start_steps) 
        policies.append(sac)

        # CHANGED DETERMNISTIC HERE
        # CHANGED ORDER OF BASELINE COLLECTION AND NORMALIZATION PARAMS
        # ADDED INITIAL STATE IN test_agent() and execute_average_policy()
        print("Test trained policy...")
        _, p_xy = sac.test_agent(T, initial_state=initial_state, deterministic=True, store_log=False, n=args.n, reset=True) # TODO: initial state seed?
        _, p_xy_non_deterministic = sac.test_agent(T, initial_state=initial_state, deterministic=False, store_log=False, n=args.n, reset=True) # TODO: initial state seed?

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        print("Executing mixed policy...")
        average_p, average_p_xy, initial_state = \
            execute_average_policy(env, policies, T, reward_fn=reward_fn, norm=normalization_factors, initial_state=initial_state, n=args.n, render=False, epoch=i)
        
        print("Calculating maxEnt entropy...")
        round_entropy = entropy(average_p.ravel())
        round_entropy_xy = entropy(average_p_xy.ravel())
        
        # Update running averages for maxEnt.
        print("Updating maxEnt running averages...")
        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_entropy/float(i+1)
        running_avg_ent_xy = running_avg_ent_xy * (i)/float(i+1) + round_entropy_xy/float(i+1)
        running_avg_p *= (i)/float(i+1) 
        running_avg_p += average_p/float(i+1)
        running_avg_p_xy *= (i)/float(i+1) 
        running_avg_p_xy += average_p_xy/float(i+1)
        
        # update reward function
        print("Update reward function")
        if args.cumulative:
            reward_fn = grad_ent(running_avg_p)
        else:
            reward_fn = grad_ent(average_p)
        average_p = None # delete big array
        
        # (save for plotting)
        running_avg_entropies.append(running_avg_ent)
        if i in indexes:
            running_avg_ps_xy.append(np.copy(running_avg_p_xy))
            print(len(running_avg_ps_xy))

        print("Collecting baseline experience....")
        p_baseline, p_baseline_xy = sac.test_agent_random(T, normalization_factors=normalization_factors, n=args.n)
        round_entropy_baseline = entropy(p_baseline.ravel())
        round_entropy_baseline_xy = entropy(p_baseline_xy.ravel())

        # Update baseline running averages.
        print("Updating baseline running averages...")
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_ent_baseline_xy = running_avg_ent_baseline_xy * (i)/float(i+1) + round_entropy_baseline_xy/float(i+1)

        running_avg_p_baseline *= (i)/float(i+1) 
        running_avg_p_baseline += p_baseline/float(i+1)
        running_avg_p_baseline_xy *= (i)/float(i+1) 
        running_avg_p_baseline_xy += p_baseline_xy/float(i+1)
        
        p_baseline = None
        
        # (save for plotting)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        if i in indexes:
            running_avg_ps_baseline_xy.append(np.copy(running_avg_p_baseline_xy))
            print(len(running_avg_ps_xy))
    
        print(average_p_xy)
        print(p_baseline_xy)
        
        # Calculate percent of state space visited.
        print("Calculate % state space reached")
        pct = np.count_nonzero(running_avg_p)/float(running_avg_p.size)
        pct_visited.append(pct)
        pct_xy = np.count_nonzero(running_avg_p_xy)/float(running_avg_p_xy.size)
        pct_visited_xy.append(pct_xy)
        
        pct_baseline = np.count_nonzero(running_avg_p_baseline)/float(running_avg_p_baseline.size)
        pct_visited_baseline.append(pct_baseline)
        pct_xy_baseline = np.count_nonzero(running_avg_p_baseline_xy)/float(running_avg_p_baseline_xy.size)
        pct_visited_xy_baseline.append(pct_xy_baseline)
        
        # Print round summary.
        col_headers = ["", "baseline", "maxEnt"]
        col1 = ["round_entropy_crop", 
                "running_avg_ent_crop", 
                "round_entropy", 
                "running_avg_ent", 
                "% state space xy", 
                "% total state space"]
        col2 = [round_entropy_baseline_xy, running_avg_ent_baseline_xy, 
                round_entropy_baseline, running_avg_ent_baseline, 
                pct_xy_baseline, pct_baseline]
        col3 = [round_entropy_xy, running_avg_ent_xy, 
                round_entropy, running_avg_ent, 
                pct_xy, pct]
        table = tabulate(np.transpose([col1, col2, col3]), col_headers, tablefmt="fancy_grid", floatfmt=".4f")
        print(table)
         
        # Plot from round.
        plotting.heatmap(running_avg_p_xy, average_p_xy, i)
        plotting.heatmap1(running_avg_p_baseline_xy, i)
        plotting.heatmap1(p_xy, i, directory='p')
        plotting.heatmap1(p_xy_non_deterministic, i, directory='p_non_deterministic')
    
    # cumulative plots.
    plotting.heatmap4(running_avg_ps_xy, running_avg_ps_baseline_xy, indexes)
    plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
    plotting.percent_state_space_reached(pct_visited, pct_visited_baseline, ext='_total')
    plotting.percent_state_space_reached(pct_visited_xy, pct_visited_xy_baseline, ext="_xy")
    
    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100, linewidth=150, precision=8)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space
    
    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    plotting.FIG_DIR = 'figs/' + args.env + '/'
    plotting.model_time = args.exp_name + '/'
    if not os.path.exists(plotting.FIG_DIR+plotting.model_time):
        os.makedirs(plotting.FIG_DIR+plotting.model_time)

    policies = collect_entropy_policies(env, args.epochs, args.T)
    env.close()

    print("*** ---------- ***")
    print("DONE")

if __name__ == "__main__":
    main()


