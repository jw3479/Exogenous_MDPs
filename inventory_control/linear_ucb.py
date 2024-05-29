import numpy as np
from scipy.stats import bernoulli
from tqdm.notebook import tqdm
import math


DEBUG = False
TRACK_MSE = False # flag for tracking the MSE for debug purposes



class UCB_Linear(object):
    '''
    Implements Algorithm 1 as described in the paper Model-Based RL with Value-Target Regression.  Note: algorithm
    assumes the costs are in the [0,1] interval. This code serves as a slight modification of the code previously
    included with their supplementary material, but extended to deal with given features \phi instead of 
    using the natural features from a tabular MDP.

    It is also flipped to deal with cost minimization rather than reward maximization
    '''

    def __init__(self, env, K, random_explore, informationBonus, d, SIMPLE_BONUS=False, conf_cons=(1/10)):
        '''
        Attributes:
            - env: simulation environment
            - K: number of episodes
            - random_explore: number of timesteps for random exploration
            - informationBonus: clipping the Q value estimates
            - d: latent dimension
            - SIMPLE_BONUS: flag for using simpler confidence parameters \beta_k
            - conf_cons : constant in front of the \beta_k confidence term
        '''
        self.env = env
        self.K = K

        self.conf_cons = conf_cons

        if not hasattr(self.env, 'states'):
            self.env.generate_states() 


        self.random_explore = random_explore
        self.SIMPLE_BONUS = SIMPLE_BONUS
        self.informationBonus = informationBonus

        self.d = d
        self.true_theta = self.env.get_theta()


        # For use in the confidence bound bonus term, see Beta function down below
        self.lam = 1.0

        #Self.L is no longer need, but will keep for now.
        self.L = 1.0
        
 
        self.M = np.identity(self.d)*self.lam # Sigma
        
        #For use in the Sherman-Morrison Update
        self.Minv = np.identity(self.d)*(1/self.lam) # Sigma inverse
        
        #See Step 2
        self.w = np.zeros(self.d) # X^T y
        
        #See Step 2
        self.theta = np.dot(self.Minv,self.w)
        
        #See Step 3
        self.delta = 1/self.K

        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): self.env.get_cost(self.env.states[s],a) for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(self.env.epLen + 1)} # self.V[env.epLen] stays zero


        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))



        if DEBUG: print(f'Finished creating Q and V matrices')


        # See Step 2, of algorithm 1
#         self.M = env.epLen**2*self.d*np.identity(self.d)
        # For use in the confidence bound bonus term, see Beta function down below
        self.lam = 1.0
        #Self.L is no longer need, but will keep for now.
        self.L = 1.0
        self.M = np.identity(self.d)*self.lam
        #For use in the Sherman-Morrison Update
        self.Minv = np.identity(self.d)*(1/self.lam)
        #See Step 2
        self.w = np.zeros(self.d)
        #See Step 2
        self.theta = np.dot(self.Minv,self.w)
        #See Step 3
        self.delta = 1/self.K
        #m_2 >= the 2-norm of theta_star, see Bandit Algorithms Theorem 20.5
        # See Theorem 20.5 for m_2



        # self.m_2 = np.linalg.norm(self.true_p) + 0.1
        self.m_2 = 1.1


#         #Initialize the predicted value of the basis models, see equation 3
#         self.X = np.zeros((env.epLen,self.d))
        #See Assumptions 2,2' and Theorem 1, this equals 1 in the tabular case
        self.C_phi = 1.0
        # See Assumption 2'(Stronger Feature Regularity), and consider the case when v_1 = v_2 = ....
        self.C_psi = np.sqrt(env.nState)
        # See Theorem 1
        self.C_M = 1.0
        # See Theorem 1
        self.C_psi_ = 1.0
        # This value scales our confidence interval, must be > 0
        self.c = 1.0
        self.d1 = env.nState * env.nAction




        if DEBUG: print(f'Finished INIT')

    def reset(self):

        #Self.L is no longer need, but will keep for now.
        self.L = 1.0
        
        self.M = np.identity(self.d)*self.lam
        
        #For use in the Sherman-Morrison Update
        self.Minv = np.identity(self.d)*(1/self.lam)
        
        #See Step 2
        self.w = np.zeros(self.d)
        
        #See Step 2
        self.theta = np.dot(self.Minv,self.w)
        
        #See Step 3
        self.delta = 1/self.K

        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): self.env.get_cost(self.env.states[s],a) for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(self.env.epLen + 1)} # self.V[env.epLen] stays zero
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

        if DEBUG: print(f'Finished creating Q and V matrices')


        # See Step 2, of algorithm 1
        # self.M = env.epLen**2*self.d*np.identity(self.d)
        # For use in the confidence bound bonus term, see Beta function down below
        self.lam = 1.0
        #Self.L is no longer need, but will keep for now.
        self.L = 1.0
        self.M = np.identity(self.d)*self.lam
        #For use in the Sherman-Morrison Update
        self.Minv = np.identity(self.d)*(1/self.lam)
        #See Step 2
        self.w = np.zeros(self.d)
        #See Step 2
        self.theta = np.dot(self.Minv,self.w)
        #See Step 3
        self.delta = 1/self.K
        # m_2 >= the 2-norm of theta_star, see Bandit Algorithms Theorem 20.5
        # See Theorem 20.5 for m_2



        # self.m_2 = np.linalg.norm(self.true_p) + 0.1
        self.m_2 = 1.1


        # Initialize the predicted value of the basis models, see equation 3
        # self.X = np.zeros((env.epLen,self.d))
        #See Assumptions 2,2' and Theorem 1, this equals 1 in the tabular case
        self.C_phi = 1.0
        # See Assumption 2'(Stronger Feature Regularity), and consider the case when v_1 = v_2 = ....
        self.C_psi = np.sqrt(self.env.nState)
        # See Theorem 1
        self.C_M = 1.0
        # See Theorem 1
        self.C_psi_ = 1.0
        # This value scales our confidence interval, must be > 0
        self.c = 1.0
        self.d1 = self.env.nState * self.env.nAction


        if TRACK_MSE:
            self.X_list = []
            self.y_list = []


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
        return sums

    def proj(self, x, lo, hi):
        '''Projects the value of x into the [lo,hi] interval'''
        return max(min(x,hi),lo)

    def update_Q(self,s,a,k,h):

        '''
        A function that updates both Q and V, Q is updated according to equation 4 and
        V is updated according to equation 2
        Inputs:
            s - the state
            a - the action
            k - the current episode
            h - the current timestep within the episode
        '''
        X = self.feature_vector(s,a,h)

        if self.SIMPLE_BONUS: # Q estimates, note here we are snapping them since the costs are all bounded
            self.Q[(h,s,a)] = self.proj(self.env.get_cost(self.env.states[s],a) + np.dot(X,self.theta) - \
            self.conf_cons * self.Beta_simple(h,k) * np.sqrt(np.dot(np.dot(np.transpose(X),self.Minv),X)), \
                            self.env.get_cost(self.env.states[s],a), self.env.epLen)
        else:
            self.Q[(h,s,a)] = self.proj(self.env.get_cost(self.env.states[s],a) + np.dot(X,self.theta) - \
            self.conf_cons * self.Beta(h,k) * np.sqrt(np.dot(np.dot(np.transpose(X),self.Minv),X)), \
                            self.env.get_cost(self.env.states[s],a), self.env.epLen)

        self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))


    def update_Qend(self,k):
        '''
        A function that updates both Q and V at the end of each episode, see step 16 of algorithm 1
        Inputs:
            k - the current episode
        '''
        #step 16

        if DEBUG: print(f'Updating Q Estimates at end of an episode!')

        for h in range(self.env.epLen-1,-1,-1): # Solving the Bellman equations
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    self.update_Q(s,a,k,h)
                self.V[(h,s)] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

        if DEBUG: print(f'Final Estimates: {self.V}')

        if k % 50 == 0:
            error = self.error(self.theta)
            print(f'Current error: {error}')
            print(f'True Theta: {self.env.true_theta} and estimate: {self.theta}')

            if TRACK_MSE:
                estimate_mse, true_mse = self.calc_mse()
                print(f'True MSE: {true_mse} and Estimate MSE: {estimate_mse}')
                assert estimate_mse <= true_mse, 'True MSE somehow lower?'


    def update_info(self, cost, s, a, s_, done, info, h, k):
        '''
        A function that performs steps 9-13 of algorithm 1
        Inputs:
            s - the current state
            a - the action
            s_ - the next state
            k - the current episode
            h - the timestep within episode when s was visited (starting at zero)
        '''
        if DEBUG: print(f'Updating information!')
        if DEBUG: print(f'Current Value: {self.V}')
        if DEBUG: print(f'{s}, {s_}')

        s = self.env.get_key(s) # gets the state keys
        s_ = self.env.get_key(s_)

        if DEBUG: print(f'{s}, {s_}')


        X = self.feature_vector(s,a,h) # gets the regression estimate V_{h+1}^k expectation over theta_hat


        y = self.V[(h+1,s_)] # gets the regression target V_{h+1}^k(s_{h+1}^k)


        if TRACK_MSE:
            self.X_list.append(X) # tracks the MSE by storing all the terms in the regression
            self.y_list.append(y)
            print(f'Covariate: {X} and value: {y}')


        self.M = self.M + np.outer(X,X)
        #Sherman-Morrison Update
        self.Minv = self.Minv - np.dot((np.outer(np.dot(self.Minv,X),X)),self.Minv) / \
                    (1 + np.dot(np.dot(X,self.Minv),X))
        #Step 13
        self.w = self.w + y*X

    def update_param(self):
        '''
        Updates our approximation of theta_star at the end of each episode, see
        Step 15 of algorithm1
        '''
        #Step 15
        self.theta = np.matmul(self.Minv,self.w) # solution to ridge regression problem = (Sigma^inverse) y


    def pick_action(self,s,h,k):
        '''
        Returns the greedy action with respect to Q_{h,k}(s,a) for a \in A
        see step 8 of algorithm 1
        Inputs:
            s - the current state
            h - the current timestep within the episode
        '''

        if DEBUG: print(f'Picking an action at episode {k} step {h} in state: {s}!')
        s = self.env.get_key(s)

        if k >= self.random_explore: # no longer doing random exploration, picks greedy wrt Q estimates
            if DEBUG: print(f'No longer random!')
            return np.argmin(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        else: # random exploration
            if DEBUG: print(f'Random at the moment')
            return np.random.choice(np.arange(self.env.nAction)) #A random policy for testing

        

    def Beta_simple(self, h, k):
        return (2*(self.env.epLen**2)*(self.d + np.log(self.K)))+((4*k*(self.env.epLen**2)) / self.K)

    def Beta(self,h,k):
        '''
        A function that return Beta_k according to Theorem 20.5 in Bandits book
        Also, if you return np.sqrt(first + second) instead of first + second then
        you get higher cumlative reward. However, in Theorem 19.2, Beta_n is already
        under the square root.
        '''
        #Step 3
        #Bonus as according to step 3
        '''
        return np.sqrt(16*pow(self.m_2,2)*pow(env.epLen,2)*self.d*np.log(1+env.epLen*k) \
            *np.log(pow(k+1,2)*env.epLen/self.delta)*np.log(pow(k+1,2)*env.epLen/self.delta))
        '''
        
        #Confidence bound from Chapter 20 of the Bandit Algorithms book, see Theorem 20.5.
        first = np.sqrt(self.lam)*self.m_2
        (sign, logdet) = np.linalg.slogdet(self.M)
        det = sign * logdet
        second = np.sqrt(2*np.log(1/self.delta) + np.log(k+1) + min(det,pow(10,10)) - np.log(pow(self.lam,self.d)))
        return first + second
        
    
    def Beta_2(self,h,k):
        '''
        Beta as defined in Mengdi's MatrixRL paper
        '''
        first = self.c*(self.C_M * self.C_psi_ ** 2)
        second = np.log(self.K*self.env.epLen*self.C_phi)*self.d1
        return h*np.sqrt(first * second)

    def calc_mse(self): # calculates the MSE in the regression between the true_theta and the estimated theta
        estimate_mse = self.lam * (np.linalg.norm(self.theta, ord=2)**2)
        true_mse = self.lam * (np.linalg.norm(self.true_theta, ord=2)**2)

        for i in range(len(self.X_list)):
            estimate_mse += (np.dot(self.X_list[i], self.theta) - self.y_list[i])**2
            true_mse += (np.dot(self.X_list[i], self.true_theta) - self.y_list[i])**2
        return (estimate_mse, true_mse)


    def name(self):
        return 'UCRL_VTR'
    
    def error(self, theta_estimate): # computes l1 error between the theta_estimate and the true_theta value
        return np.linalg.norm(theta_estimate - self.true_theta, ord = 1)