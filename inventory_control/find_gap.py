import numpy as np
import pandas as pd
import base_stock
import simulator
import matplotlib.pyplot as plt
import helper
import seaborn as sns
import online_convex_optimization
import config_list
import opt_policy

'''
File to loop over different parameters in the config to check the optimality gap
between best base stock policy and optimal policy=
'''

# mean_list = [1,2,3,4,5,6]
mean_list = [3]
# c_list = [1, 2, 3, 4, 5, 6]
c_list = [6,5,4,3,2,1]
h_list = [1, 2, 3, 4, 5, 6]

max_gap = -1
for mu in mean_list:
    for c in c_list:
        for h in h_list:
            print(f'Evaluating: mu: {mu}, c: {c}, h: {h}')
            config = config_list.get_medium_config(mu, c, h)
            inv_env = simulator.InventoryEnvironment(config = config)

            opt_base_stock, min_base_cost = helper.get_optimal_base_stock(inv_env, int(1e4), low = 0, high = config['max_order'])
            opt_algo = opt_policy.Optimal_Policy(inv_env, config['max_order']+1)
            opt_value = opt_algo.V[0, inv_env.get_key(inv_env.starting_state)]
            opt_value = opt_value
            print(f'Optimal Cost: {opt_value} and Optimal Base Stock: {min_base_cost}')

            min_base_cost = max(opt_value, min_base_cost) # for numerical precision since base stock is evaluated with monte carlo estimates

            if min_base_cost - opt_value >= max_gap:
                print(f'Updating Best Gap!')
                max_gap = min_base_cost - opt_value
                best_c = c
                best_mu = mu
                best_h = h

print(f'Maximum gap achieved at: mu : {mu}, c: {c}, h: {h}')




