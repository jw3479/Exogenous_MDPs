import numpy as np
from scipy.stats import bernoulli
from tqdm.notebook import tqdm
import math


DEBUG = False

class Random_Policy(object):
    '''
        Executes a randomized policy
    '''
    
    def __init__(self, env):
        
        self.env = env

    def reset(self):
        return

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
        Returns a random action
        '''
        
        return np.random.choice(np.arange(self.env.nAction))
        

    def name(self):
        return 'Optimal Policy'
