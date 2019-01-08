# self.sim.data.qpos are the positions, with the first 7 element the 
# 3D position (x,y,z) and orientation (quaternion x,y,z,w) of the torso, 
# and the remaining 8 positions are the joint angles.

# The [2:], operation removes the first 2 elements from the position, 
# which is the X and Y position of the agent's torso.

# self.sim.data.qvel are the velocities, with the first 6 elements 
# the 3D velocity (x,y,z) and 3D angular velocity (x,y,z) and the 
# remaining 8 are the joint velocities.

# 0 - x position
# 1 - y position
# 2 - z position
# 3 - x torso orientation
# 4 - y torso orientation
# 5 - z torso orientation
# 6 - w torso orientation
# 7-14 - joint angles

# 15-21 - 3d velocity/angular velocity
# 23-29 - joint velocities

import gym
import time
import numpy as np

import utils
args = utils.get_args()

env = gym.make('Ant-v2')

dim_dict = {
    0:"x",
    1:"y",
    2:"z",
    3:"x torso",
    4:"y torso",
    5:"z torso",
    6:"w torso",
    7:"joint 1 angle",
    8:"joint 2 angle",
    9:"joint 3 angle",
    10:"joint 4 angle",
    11:"joint 5 angle",
    12:"joint 6 angle",
    13:"joint 7 angle",
    14:"joint 8 angle",
    15:"3d veloity/angular velocity",
    16:"3d veloity/angular velocity",
    17:"3d veloity/angular velocity",
    18:"3d veloity/angular velocity",
    19:"3d veloity/angular velocity",
    20:"3d veloity/angular velocity",
    21:"3d veloity/angular velocity",
    22:"3d veloity/angular velocity",
    23:"joint velocity",
    24:"joint velocity",
    25:"joint velocity",
    26:"joint velocity",
    27:"joint velocity",
    28:"joint velocity",
    29:"joint velocity",
}

qpos = env.env.init_qpos
qvel = env.env.init_qvel

state_dim = int(env.env.state_vector().shape[0])
action_dim = int(env.action_space.sample().shape[0])

features = [2,7,8,9,10]
min_bin = -3
max_bin = 3
height_bins = 20
num_bins = 20

start = 3
stop = 5
num_bins_full = 10

reduce_dim = args.reduce_dim
G = np.transpose(np.random.normal(0, 1, (state_dim, reduce_dim)))

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    state_bins = [
        # height
        discretize_range(0.2, 1.0, height_bins),
        # other fields
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins)
    ]
    return state_bins

def get_state_bins_reduced():
    state_bins = []
    for i in range(reduce_dim):
        state_bins.append(discretize_range(min_bin, max_bin, num_bins))
    return state_bins

def get_state_bins_full_state():
    state_bins = []
    for i in range(start, stop):
        state_bins.append(discretize_range(-3, 3, num_bins_full))
    return state_bins


def get_num_states(state_bins):
    num_states = []
    for i in range(len(state_bins)):
        num_states.append(len(state_bins[i]) + 1)
    return num_states

state_bins = []
if args.gaussian:
    state_bins = get_state_bins_reduced()
else:
    state_bins = get_state_bins()
num_states = get_num_states(state_bins)

state_bins_full = get_state_bins_full_state()
num_states_full = tuple([num_bins_full for i in range(start, stop)])

# Discretize the observation features and reduce them to a single list.
def discretize_state_full(observation):
    state = []
    for i in range(start, stop):
        feature = observation[i]
        state.append(discretize_value(feature, state_bins_full[i - start]))
    return state
# Goal: discretize the state 

def discretize_state_normal(observation):
    state = []
    for i, idx in enumerate(features):
        state.append(discretize_value(observation[idx], state_bins[i]))
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state_reduced(observation):
    # print(observation)
    observation = np.dot(G, observation)
    # print(observation)
    state = []
    for i, feature in enumerate(observation):
        state.append(discretize_value(feature, state_bins[i]))
    return state

# Discretize the observation features and reduce them to a single list.
def discretize_state(observation):
    if args.gaussian:
        state = discretize_state_reduced(observation)
    else:
        state = discretize_state_normal(observation)

    return state

def get_height_dimension(arr):
    return np.array([np.sum(arr[i]) for i in range(arr.shape[0])])

def get_ith_dimension(arr, i):
    return np.array([np.sum(arr[j]) for j in range(arr.shape[i])])


