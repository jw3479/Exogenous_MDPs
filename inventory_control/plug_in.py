import numpy as np
from scipy.stats import bernoulli
from tqdm.notebook import tqdm
import math


DEBUG = False
TRUE_THETA = False



class PlugIn_Policy(object):
    '''
        Algorithm following the plug in approach where we use the estimated PXi from observed
        samples and solves the empirical Bellman equations with that estimate for PXi

    '''

    def __init__(self, env, d, bias_flag):
        '''
            Takes as input the environment, dimension d, as well as bias_flag

            If bias_flag = True, in the inventory setting we update the PXi estimated by min(x, d) versus
            the true demand value (censored demand occurs due to lost sales)
        '''
        
        self.env = env
        self.d = d
        self.bias_flag = bias_flag

        if not hasattr(self.env, 'states'):
            self.env.generate_states()


        self.true_theta = self.env.get_theta()

        self.counts = np.zeros(self.d)
        self.theta_hat = np.ones(self.d)/self.d # estimate for PXi
        self.num_samples = 0 # number of samples in the empirical estimate of theta and the estimate itself

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
                    X = self.feature_vector(s,a,h)
                    self.Q[(h,s,a)] = self.env.get_cost(self.env.states[s],a) + np.dot(X,self.true_theta)

                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def reset(self):
        '''
        Resets the estimate of theta hat back to original uniform distribution
        '''

        self.counts = np.zeros(self.d)
        self.theta_hat = np.ones(self.d)/self.d
        self.num_samples = 0 # number of samples in the empirical estimate of theta and the estimate itself

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
                    X = self.feature_vector(s,a,h)
                    self.Q[(h,s,a)] = self.env.get_cost(self.env.states[s],a) + np.dot(X,self.true_theta)

                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))



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
        # Solving the Bellman equations
        if DEBUG: print(f'Solving Bellman Equations!')



        # Updating estimate for theta_hat
        self.theta_hat = self.counts / (self.num_samples)

        if DEBUG:
            print(f'Timestep: {k}, num_samples: {self.num_samples} and counts: {self.counts}')
            print(f'Sum of counts: {np.sum(self.counts)}')
            print(f'Sum of theta_hat: {np.sum(self.theta_hat)}')

        assert np.abs(np.sum(self.theta_hat) - 1) <= 0.05, 'Not a valid distribution for theta hat estimation'

        if DEBUG: print(f'Theta Hat: {self.theta_hat} and true theta: {self.true_theta}')
        if DEBUG: print(f'Current error: {np.linalg.norm(self.theta_hat - self.true_theta, ord=1)}')

        if TRUE_THETA: self.theta_hat = self.true_theta


        for h in range(self.env.epLen-1,-1,-1): # Solves Bellman with estimated PXi
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    X = self.feature_vector(s,a,h)
                    self.Q[(h,s,a)] = self.env.get_cost(self.env.states[s],a) + np.dot(X,self.true_theta)

                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))


    def update_info(self, cost, s, a, s_, done, info, h, k):
        '''
        Updates the estimated theta_hat using an additional sample from info
        '''

        if self.bias_flag: # If we are inluding bias, only update based on sales
            current_xi = min(self.env.max_order, info['sales'])
        else:
            current_xi = min(self.env.max_order, info['true_demand']) # otherwise updated based on true demand

        self.counts[current_xi] += 1
        self.num_samples += 1

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
