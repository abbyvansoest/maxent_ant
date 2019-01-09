import numpy as np
import ant_utils
import time

class ExperienceBuffer:
    
    def __init__(self):
        self.buffer = []
        self.normalization_factors = []
        self.normalized = False
        self.p = None
        self.p_full_dim = None
        
    def store(self, state):
        if self.normalized:
            raise ValueError("ERROR! Do not store in buffer after normalizing.")
        # TODO: check dimension of obs: should be dimension of ant env.env.state_vector()
        state = np.dot(ant_utils.G, state)
        self.buffer.append(state)
    
    def normalize(self):
        # for each index in reduced dimension,
        # find the largest value in self.buffer
        # divide all elements at that index 
        for i in range(ant_utils.reduce_dim):
            i_vals = [x[i] for x in self.buffer]
            max_i_val = max(i_vals)
            self.normalization_factors.append(max_i_val)
            for obs in self.buffer:
                obs[i] = obs[i]/max_i_val
        
        # you don't need to do this again.
        self.normalized = True
        return self.buffer
    
    def get_discrete_distribution(self):
        
        if self.p is not None:
            return self.p
        
        # normalize buffer experience
        if not self.normalized:
            self.normalize()
            
        p = np.zeros(shape=(tuple(ant_utils.num_states)))
        for obs in self.buffer:
            # discritize obs, add to distribution tabulation.
            p[tuple(ant_utils.discretize_state(obs))] += 1
        
        p /= len(self.buffer)
        self.p = p
            
        return p
        
    def get_discrete_distribution_full(self):
        
        if self.p_full_dim is not None:
            return self.p_full_dim
        
        # normalized buffer experience
        if not self.normalized:
            self.normalize()
            
        p_full_dim = np.zeros(shape=(ant_utils.num_states_full))
        for obs in self.buffer:
            # discritize obs, add to distribution tabulation.
            p_full_dim[tuple(ant_utils.discretize_state_full(obs))] += 1
        
        p_full_dim /= len(self.buffer)
        self.p_full_dim = p_full_dim
                
        return p_full_dim
