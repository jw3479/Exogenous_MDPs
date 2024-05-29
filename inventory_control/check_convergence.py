import numpy as np
import pandas as pd
import base_stock
import simulator
import matplotlib.pyplot as plt
import helper
import seaborn as sns
import online_convex_optimization
import tabular_ucb
import linear_ucb
import opt_policy
import config_list
from tqdm import tqdm

DEBUG = False
seed = 5
config = config_list.config_small_lower_distribution # configuration of simulator to run on




np.random.seed(5) # setting seed for simulation

### SETTING UP PARAMETERS FOR THE EXPERIMENT

num_episodes = 125 # K


# Setting up the inventory environment
inv_env = simulator.InventoryEnvironment(config = config)

algo = linear_ucb.UCB_Linear(env=inv_env, K = num_episodes, random_explore = 100, informationBonus = 10, d = config['max_order'] + 1)

data_dict = []



# GET THE OPTIMAL Q VALUES

opt_algo = opt_policy.Optimal_Policy(inv_env, config['max_order']+1)
opt_estimates = opt_algo.Q
opt_estimates['episode'] = 'optimal'

data_dict.append(opt_estimates)

# for k in range(num_episodes): # number of episodes K
for k in range(num_episodes):

    if DEBUG and k % 50 == 0:
        print(f'Starting episode: {k}') # tracker for progress within the episode

    state = inv_env.reset() # initialize the environment

    for step in range(config['epLen']):
        action = algo.pick_action(state, step, k) # selects action according to the algoirthm

        cost, new_state, done, info = inv_env.step(action) # puts action into environment and get next state

        algo.update_info(cost, state, action, new_state, done, info, step, k) # "updates information" for the algorithm based on one sample
        state = new_state

    print(f'Finished an episode!')
    algo.update_param() # update internal parameters for the algorithms
    algo.update_Qend(k)

    current_estimates = dict(algo.Q)
    current_estimates['episode'] = k

    data_dict.append(current_estimates)



df = pd.DataFrame.from_dict(data_dict) # generates a data frame from the data
log_file = './data/check_convergence_inv_data_'+str(seed)+'_'+config['name']+'.csv'
df.to_csv(log_file) # saves the data to a csv












