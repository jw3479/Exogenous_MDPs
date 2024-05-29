import numpy as np
from scipy.stats import bernoulli
from tqdm.notebook import tqdm



DEBUG = False

class UCB_Tabular(object):
    '''
    Algorithm 1 as described in the paper Model-Based RL with
    Value-Target Regression
    The algorithm assumes that the rewards are in the [0,1] interval.
    '''


    def __init__(self,env,K,random_explore,informationBonus):
        
        self.env = env
        self.K = K

        if not hasattr(self.env, 'states'):
            self.env.generate_states()


        # A unit test that randomly explores for a period of time then learns from that experience
        # Here self.random_explore is a way to select a period of random exploration.
        # When the current episode k > total number of episodes K divided by self.random_explore
        # the algorithm switches to the greedy action with respect to its action value Q(s,a).
        
        self.random_explore = random_explore

        self.informationBonus = informationBonus

        # Here the dimension (self.d) for the Tabular setting is |S x A x S| as stated in Appendix B
        self.d = self.env.nState * self.env.nAction * self.env.nState

        # In the tabular setting the basis models is just the dxd identity matrix, see Appendix B
        self.P_basis = np.identity(self.d)


        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(env.epLen + 1)} # self.V[env.epLen] stays zero

        if DEBUG: print(f'Finished creating Q and V matrices')

        #self.create_value_functions()
        #The index of each (s,a,s') tuple, see Appendix B
        self.sigma = {}
        self.state_idx = {}
        self.createIdx()

        if DEBUG: print(f'Finished creating Idx')


        #See Step 2, of algorithm 1
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

        self.P_basis = np.identity(self.d)

        #Our Q-values are initialized as a 2d numpy array, will eventually convert to a dictionary
        self.Q = {(h,s,a): 0.0 for h in range(self.env.epLen) for s in self.env.states.keys() \
                   for a in range(self.env.nAction)}
        #Our State Value function is initialized as a 1d numpy error, will eventually convert to a dictionary
        self.V = {(h,s): 0.0 for s in self.env.states.keys() for h in range(self.env.epLen + 1)} # self.V[env.epLen] stays zero


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
        self.C_psi = np.sqrt(self.env.nState)
        # See Theorem 1
        self.C_M = 1.0
        # See Theorem 1
        self.C_psi_ = 1.0
        # This value scales our confidence interval, must be > 0
        self.c = 1.0
        self.d1 = self.env.nState * self.env.nAction





    def feature_vector(self,s,a,h):
        '''
        Returning sum_{s'} V[h+1][s'] P_dot(s'|s,a),
        with V stored in self.
        Inputs:
            s - the state
            a - the action
            h - the current timestep within the episode
        '''
        sums = np.zeros(self.d)
        # print(self.V.shape)
        # print(self.P_basis.shape)
        # print(self.sigma.shape)

        for s_ in self.env.states.keys():
            # print(self.V[h+1, s_])
            # # print(self.sigma[(s,a,s_)])
            # x = self.V[h+1, s_]
            # y = self.sigma[(s,a,s_)]
            # z = self.P_basis[y]
            sums += self.V[h+1,s_] * self.P_basis[self.sigma[(s,a,s_)]]
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
        Currently, does not properly compute the Q-values but it does seem to learn theta_star
        '''

        #Here env.R[(s,a)][0] is the true reward from the environment
        # Alex's code: X = self.X[h,:]
        # Suggested code:
        X = self.feature_vector(s,a,h)
        self.Q[h,s,a] = self.proj(self.env.get_reward(self.env.states[s],a) + np.dot(X,self.theta) + self.Beta(h,k) \
            # Adding + 10 to the epLen seems to allow for less information loss since the bound does not get forgotten
            * np.sqrt(np.dot(np.dot(np.transpose(X),self.Minv),X)), 0, self.env.epLen + self.informationBonus )
        self.V[h,s] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))

    def update_Qend(self,k):
        '''
        A function that updates both Q and V at the end of each episode, see step 16 of algorithm 1
        Inputs:
            k - the current episode
        '''
        #step 16

        if DEBUG: print(f'Updating Q Estimates at end of an episode!')
        for h in range(self.env.epLen-1,-1,-1):
            for s in self.env.states.keys():
                for a in range(self.env.nAction):
                    #Here env.R[(s,a)][0] is the true reward from the environment
                    # Alex's code: X = self.X[h,:]
                    # Suggested code:
                    self.update_Q(s,a,k,h)
                self.V[h,s] = min(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))


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

        # print(f'{s}, {s_}')

        s = self.env.get_key(s)
        s_ = self.env.get_key(s_)


        #Step 10
#         self.X[h,:] = self.feature_vector(s,a,h) # do not need to store this
        X = self.feature_vector(s,a,h)
        #Step 11
        y = self.V[h+1,s_]
#         if s_ != None:
#             y = self.V[h+1][s_]
#         else:
#             y = 0.0
        #Step 12
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
        #print(self.M)
        self.theta_old = self.theta
        self.theta = np.matmul(self.Minv,self.w)

    def pick_action(self,s,h,k):
        '''
        Returns the greedy action with respect to Q_{h,k}(s,a) for a \in A
        see step 8 of algorithm 1
        Inputs:
            s - the current state
            h - the current timestep within the episode
        '''
        #step 8
        if DEBUG: print(f'Picking an action at episode {k} step {h} in state: {s}!')
        s = self.env.get_key(s)
        if k >= self.K /self.random_explore:
            if DEBUG: print(f'No longer random!')
            #print (max(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)])))
            return np.argmin(np.array([self.Q[(h,s,a)] for a in range(self.env.nAction)]))
        else:
            if DEBUG: print(f'Random at the moment')
            return np.random.choice(np.arange(self.env.nAction)) #A random policy for testing

    def createIdx(self):
        '''
        A simple function that creates sigma according to Appendix B.
        Here sigma is a dictionary who inputs is a tuple (s,a,s') and stores
        the interger index to be used in our basis model P.
        '''
        i = 0
        j = 0
        k = 0
        for s in self.env.states.keys():
            self.state_idx[s] = int(j)
            j += 1
            for a in range(self.env.nAction):
                for s_ in self.env.states.keys():
                    self.sigma[(s,a,s_)] = int(i)
                    i += 1
        
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
        #second = np.sqrt(2*np.log(1/self.delta) + self.d*np.log((self.d*self.lam + k*self.L*self.L)/(self.d*self.lam)))
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

    def run(self):
        R = []
        reward = 0.0
        for k in tqdm(range(1,self.K+1)):
            self.env.reset()
            done = 0
            while done != 1:
                s = self.env.state
                h = self.env.timestep
                a = self.act(s,h,k)
                r,s_,done = self.env.advance(a)
                reward += r
                self.update_stat(s,a,s_,h)
            self.update_param()
            self.update_Qend(k)
            R.append(reward)
        return R

    def name(self):
        return 'UCRL_VTR'
    