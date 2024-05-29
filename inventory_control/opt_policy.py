import numpy as np
from scipy.stats import bernoulli
from tqdm.notebook import tqdm
import math


DEBUG = False

class Optimal_Policy(object):
    '''
        Solves the full Bellman equations based on PXi and evaluates it
    '''

    def __init__(self, env, d):
        
        self.env = env # stores environment and d
        self.d = d


        if not hasattr(self.env, 'states'):
            self.env.generate_states()


        self.true_theta = self.env.get_theta() # gets true PXi

        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(self.env.epLen + 1)} # self.V[env.epLen] stays zero


        # Solving the Bellman equations
        if DEBUG: print(f'Solving Bellman Equations!')

        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    X = self.feature_vector(s,a,h) # Expected V_{h+1}

                    self.Q[(h,s,a)] = self.env.get_cost(self.env.states[s],a) + np.dot(X,self.true_theta)

                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))


    def reset(self):
        return


    def feature_vector(self,s,a,h):
        '''
        Returning sum_{s'} V[h+1][s'] \phi(s'|s,a),
        with V stored in self.
        Inputs:
            s - the state
            a - the action
            h - the current timestep within the episode
        '''
        sums = np.zeros(self.d)
        for s_ in self.env.states.keys():
            sums += self.V[(h+1,s_)] * self.env.feature_vector(self.env.states[s],a,self.env.states[s_])
        # print(f'Final Sum: {sums}')
        return sums


    def update_Q(self,s,a,k,h):
        return

    def update_Qend(self,k):
        return

    def update_info(self, cost, s, a, s_, done, info, h, k):
        return

    def update_param(self):
        return

    def pick_action(self,s,h,k):
        '''
        Returns the greedy action with respect to Q_{h,k}(s,a) for a \in A
        '''
        s = self.env.get_key(s)

        return np.argmin(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        

    def name(self):
        return 'Optimal Policy'