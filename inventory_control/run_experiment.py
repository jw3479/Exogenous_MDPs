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
import plug_in
import opt_policy
import config_list
import random_policy
from tqdm import tqdm

'''
Runs an experiment comparing different policy performance and saves their output
'''

DEBUG = True # DEBUGS The experiment run
SAVE = False # Flag for saving data
seed = 5 # seed for the randomized demand

np.random.seed(5) # setting seed for simulation

### SETTING UP PARAMETERS FOR THE EXPERIMENT

num_simulations = 1 # for expected regret
num_episodes = 1000 # 

'''
Picking out a configuration for the experiment
'''
# config = config_list.get_small_config(mean) # configuration of simulator to run on
config = config_list.get_no_lead_config(7, c=6, h=4, support=8)
# config = config_list.get_medium_config(3, 6, 1)



# Setting up the inventory environment
inv_env = simulator.InventoryEnvironment(config = config)

print(f'Distribution: {inv_env.demand_dist}')

# List of algorithms to run, comment out appropriately
algo_list = {
    # 'random' : random_policy.Random_Policy(env=inv_env),
    # 'linear_ucb_1_10' : linear_ucb.UCB_Linear(env=inv_env, K = num_episodes, random_explore = 100, informationBonus = 10, d = config['max_order'] + 1, conf_cons=(1/10)),
    # 'linear_ucb_1' : linear_ucb.UCB_Linear(env=inv_env, K = num_episodes, random_explore = 100, informationBonus = 10, d = config['max_order'] + 1, conf_cons=(1)),
    # 'linear_ucb_10' : linear_ucb.UCB_Linear(env=inv_env, K = num_episodes, random_explore = 100, informationBonus = 10, d = config['max_order'] + 1, conf_cons=(10)),
    # 'online_base_stock' : online_convex_optimization.online_convex_opt_algo(num_episodes, config['epLen'], config['max_order'], conf_cons = 1),
    # 'online_base_stock_1_10' : online_convex_optimization.online_convex_opt_algo(num_episodes, config['epLen'], config['max_order'], conf_cons = 1/10),
    # 'online_base_stock_1_100' : online_convex_optimization.online_convex_opt_algo(num_episodes, config['epLen'], config['max_order'], conf_cons = 1/100),
    'online_base_stock_1_50' : online_convex_optimization.online_convex_opt_algo(num_episodes, config['epLen'], config['max_order'], conf_cons = 1/50),
    # 'online_base_stock_1_200' : online_convex_optimization.online_convex_opt_algo(num_episodes, config['epLen'], config['max_order'], conf_cons = 1/200),
    # 'online_base_stock_1_300' : online_convex_optimization.online_convex_opt_algo(num_episodes, config['epLen'], config['max_order'], conf_cons = 1/300),
    # 'true_plug_in' : plug_in.PlugIn_Policy(env = inv_env, d = config['max_order'] + 1, bias_flag = False),
    # 'bias_plug_in' : plug_in.PlugIn_Policy(env = inv_env, d = config['max_order'] + 1, bias_flag = True),
    }


# if DEBUG: # prints starting interval and optimal base stock level
#     print(f'Original Base Stock Interval: {0, config["max_order"]}')
#     opt_base_stock, min_cost = helper.get_optimal_base_stock(inv_env, int(1e4), low = 0, high = config['max_order'], tol=1)
#     print(f"Optimal Base Stock Level: {opt_base_stock} and its cost: {min_cost}")


for algo_name, algorithm in algo_list.items(): # loops over all of the algorithms
    print(f'Running for algorithm: {algo_name}')
    data = []
    for i in range(num_simulations): # each algorithm gets run multiple times to plot expected regret
        
        print(f'Iteration: {i}')

        algorithm.reset() # resets the algorithm so not carrying over data from the previous iteration

        for k in range(num_episodes): # number of episodes K

            if DEBUG and k % 100 == 0:
                print(f'Starting episode: {k}') # tracker for progress within the episode

            state = inv_env.reset() # initialize the environment

            for step in range(config['epLen']):
                action = algorithm.pick_action(state, step, k) # selects action according to the algoirthm

                cost, new_state, done, info = inv_env.step(action) # puts action into environment and get next state

                algorithm.update_info(cost, state, action, new_state, done, info, step, k) # "updates information" for the algorithm based on one sample
                state = new_state
                data_dict = {'algorithm' : algo_name, 'iteration' : i, 'episode': k, 'step' : step, 'cost' : cost}
                data.append(data_dict)
            algorithm.update_param() # update internal parameters for the algorithms
            algorithm.update_Qend(k)

        if algo_name == 'online_base_stock':
            print(f'Final Base Stock Interval: {algorithm.low, algorithm.high}')

        if algo_name in ['true_plug_in', 'bias_plug_in']:
            print(f'Final true theta: {algorithm.true_theta} and theta hat: {algorithm.theta_hat}')
            print(f'Final error: {np.linalg.norm(algorithm.true_theta - algorithm.theta_hat, ord=1)}')

    df = pd.DataFrame.from_dict(data) # generates a data frame from the data
    log_file = './data/inv_data_'+str(seed)+'_'+str(algo_name)+'_'+config['name']+'.csv'
    if SAVE: df.to_csv(log_file) # saves the data to a csv

print(f'Finished!')
