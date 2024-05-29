import numpy as np
import sys
import itertools
import random


DEBUG = False

class InventoryEnvironment():
    """

    Environment for a single retailer inventory control model with positive lead time and lost sales. The goal is to make
    ordering decisions online to minimze cumulative cost.  

    Attributes:
        lead_times: int representing the lead times of each supplier.
        sales_penalty: penalty for lost sales
        hold_cost: holding cost
        epLen:  number of time steps in the simulation
        max_order: maximum order quantity
        max_inventory: maximum inventory level
        timestep: current timestep
        starting_state: starting state
    """

    
    def __init__(self, config):
        """
        Args:
            config: A dictionary containt the following parameters required to set up the environment:
                lead_times: int representing the lead times of each supplier.
                sales_penalty: penalty for lost sales
                holding_cost: holding cost
                epLen:  number of time steps in the simulation
                demand_dist : function which returns the random variable corresponding to the demand
                max_order: maximum order quantity
                max_inventory: maximum inventory level
                starting_state: starting state for inventory
            """
        self.config = config
        
        self.lead_time = config['lead_time']
        self.sales_penalty = config['sales_penalty']
        self.holding_cost = config['holding_cost']
        self.epLen = config['epLen'] # H
        self.demand_dist = config['demand_dist'] # PXi
        self.true_theta = self.demand_dist # stores into true_theta
        self.max_order = config['max_order'] # maximum value of a
        self.max_inventory = config['max_inventory'] # maximum value of state
        self.starting_state = config['starting_state']
        self.R = {} # empty dictionary for the reward
        self.state = np.asarray(self.starting_state) # starting state


        self.cost_norm_factor = self.holding_cost * self.max_inventory + self.sales_penalty * self.max_order # normalizing factor
                # so that the costs are in [0,1]

        self.d = self.max_order + 1 # dimension is possible demand values

        self.nState = (self.max_inventory+1) ** (self.lead_time + 1) # stores the number of discrete states and actions
        self.nAction = self.max_order + 1

        if DEBUG: print(f'Number of states: {self.nState} and actions: {self.nAction}')


        assert len(self.state) == self.lead_time + 1, "Starting state needs to be correct dimension"

        self.timestep = 0

    def get_cost(self, s, a):
        '''
            Returns an estimate of R(s,a) using monte-carlo simulation
        '''

        if (s,a) in self.R.keys():
            return self.R[(s,a)] # checks to see if R(s,a) has already been stored and return it

        else: # otherwise, compute it PXi and add to dictionary
            
            weighted_cost = 0
            for demand in range(self.max_order + 1):

                if self.lead_time > 0: # then there is new outstanding inventory that adds onto the on hand inventory
                    on_hand = s[0] + s[1] # updating on hand inventory
                
                else:
                    on_hand = s[0] + a # if no lead time, new state is just updated by the ordering amount
        
                sales = np.minimum(on_hand, demand) # sales is the minimum of on-hand inventory and demand
    
                new_on_hand = min(self.max_inventory, on_hand - sales) # updating new on-hand inventory for the next timestep

                cost = ( self.holding_cost * new_on_hand + self.sales_penalty * np.maximum(demand - on_hand, 0)) / self.cost_norm_factor
                
                weighted_cost += self.demand_dist[demand] * cost

            self.R[(s,a)] = weighted_cost

            assert self.R[(s,a)] >= 0, 'Reward estimate is negative'
            assert self.R[(s,a)] <= 1, 'Reward estimate is larger than one'
            
            
            return self.R[(s,a)]

    def get_theta(self): # gets the true PXi
        return self.demand_dist


    def generate_states(self): # calls the inventorment to store the set of states
        print(f'Generating states for the simulator')
        self.states = self.generate_vectors(self.max_inventory, self.lead_time)
        assert len(self.states) == self.nState, 'Mismatch number of states'
        print(f'Finished!')


    def generate_vectors(self, B, L): # generates the state vectors of [B]^{L+1}
        # Create the range of values each component of the vector can take
        value_range = range(B + 1)
        
        # Create an iterator for all possible vectors of length L+1
        vector_iterator = itertools.product(value_range, repeat=L+1)
        
        # Initialize the dictionary to store vectors with their indices
        vector_dict = {}
        
        # Enumerate over the iterator to fill the dictionary with index, vector pairs
        for index, vector in enumerate(vector_iterator):
            vector_dict[index] = vector
        
        return vector_dict


    def get_key(self, val): # gets the index for a corresponding state = val
        for key, value in self.states.items():
            # print(value)
            if list(value) == list(val):
                return key
        return None  # Return None if no key with that value is found



    def feature_vector(self, state, action, new_state):
        # Returns the d - dimensional feature vector for the given state, action, new_state pair

        feature_vec = np.zeros(self.d) # initializes the vector

        for demand in range(self.max_order+1): # loops over possible demand values
            newState = np.copy(state)

            if self.lead_time > 0: # then there is new outstanding inventory that adds onto the on hand inventory
                newState[0] = state[0] + state[1] # updating on hand inventory
            
            else:
                newState[0] = state[0] + action # if no lead time, new state is just updated by the ordering amount
        
            on_hand = newState[0]
            sales = np.minimum(on_hand, demand) # sales is the minimum of on-hand inventory and demand

            newState[0] = min(self.max_inventory, on_hand - sales) # updating new on-hand inventory for the next timestep

            if self.lead_time > 0:
                newState[1:] = np.append(state[2:], action)
                
            if np.all(newState == new_state): # if s' = f(s,a,\xi) then feature_vec is 1, otherwise zero
                feature_vec[demand] = 1

        # if DEBUG: print(f'Feature Vector: {feature_vec}')
        return feature_vec
    
    def get_config(self):
        return self.config
    
    
    def seed(self, seed=None):
        """Sets the numpy seed to the given value

        Args:
            seed: The int represeting the numpy seed."""
        np.random.seed(seed)
        return seed


    def reset(self):
        """Reinitializes variables and returns the starting state."""
        self.state = np.asarray(self.starting_state)
        self.timestep = 0
        return self.state
    

    def step_from_state(self, state, action):
        """
        Move one step in the environment assuming the current state is state.
        NOTE: Does not update the INTERNAL state of the simulator.  Only returns it in the fashion of a generative model.

        Args:
            state: current state for the system
            action: ordering amount from the supplier

        Returns:
            reward : reward based on action chosen
            newState : list representing the new state of environment after the action
            done : boolean flag indicating end of the episode
            info : dictionary containing the sales (censored information)
        """

        assert action <= self.max_order, 'Action more than max order'
        assert np.all(state <= self.max_inventory), 'State more than max inventory'

        demand = np.random.choice(a = range(self.max_order + 1), size = 1, p = self.demand_dist)[0] # stores empirical demand distribution
        newState = np.copy(state) # setting up the newState to be same as the old state

        if self.lead_time > 0: # then there is new outstanding inventory that adds onto the on hand inventory
            newState[0] = state[0] + state[1] # updating on hand inventory
        
        else:
            newState[0] = state[0] + action # if no lead time, new state is just updated by the ordering amount
        
        on_hand = newState[0]
        sales = np.minimum(on_hand, demand) # sales is the minimum of on-hand inventory and demand

        newState[0] = min(self.max_inventory, on_hand - sales) # updating new on-hand inventory for the next timestep

        cost = ( self.holding_cost * newState[0] + self.sales_penalty * np.maximum(demand - on_hand, 0)) / self.cost_norm_factor
        # NOTE: Normalizing the cost here

        if self.lead_time > 0:
            newState[1:] = np.append(state[2:], action)
        

        info = {'sales' : sales, 'true_demand' : demand}


        return cost, newState, False, info



    def step(self, action):
        """
        Move one step in the environment.

        Args:
            action: ordering amount from the supplier

        Returns:
            reward : reward based on action chosen
            newState : list representing the new state of environment after the action
            done : boolean flag indicating end of the episode
            info : dictionary containing the sales (censored information)
        """
        
        assert action <= self.max_order, 'Action more than max order'
        assert np.all(self.state <= self.max_inventory), 'State more than max inventory'


        demand = np.random.choice(a = range(self.max_order + 1), size = 1, p = self.demand_dist)[0] # stores empirical demand distribution

        state = self.state

        newState = np.copy(state) # setting up the newState to be same as the old state

        if self.lead_time > 0: # then there is new outstanding inventory that adds onto the on hand inventory
            newState[0] = state[0] + state[1] # updating on hand inventory
        
        else:
            newState[0] = state[0] + action # if no lead time, new state is just updated by the ordering amount
        
        on_hand = newState[0]
        sales = np.minimum(on_hand, demand) # sales is the minimum of on-hand inventory and demand

        newState[0] = min(self.max_inventory, on_hand - sales) # updating new on-hand inventory for the next timestep
            # if on-hand is larger than max inventory, throw it out!
       
        cost = ( self.holding_cost * newState[0] + self.sales_penalty * np.maximum(demand - on_hand, 0)) / self.cost_norm_factor
        # NOTE: Normalizing the cost here
        
        if self.lead_time > 0:
            newState[1:] = np.append(state[2:], action) # shifting over the remaining orders

        self.state = newState
        self.timestep += 1

        if self.timestep == self.epLen:
            done = True
        else:
            done = False

        info = {'sales' : sales, 'true_demand' : demand}
        if DEBUG: print(f'Old State: {state}, action : {action}, demand: {demand}, sales : {sales}, newState : {newState}, cost: {cost}')
        return cost, newState, done, info