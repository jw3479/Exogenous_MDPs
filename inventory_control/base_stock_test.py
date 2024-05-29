import numpy as np
import pandas as pd
import base_stock
import simulator
import matplotlib.pyplot as plt

import seaborn as sns
import config_list

plt.style.use('PaperDoubleFig.mplstyle.txt')


# Python file which evaluates different base stock policies

print('#######################')
print('Creating Environment')

### SETTING UP PARAMETERS FOR THE EXPERIMENT

num_simulations = 100000

config = config_list.get_no_lead_config(4, 6, 4, 10) # picking a configuration for the simulation

eps = .5 # discretization parameter for evaluating the base stock levels

# Setting up the inventory environment
inv_env = simulator.InventoryEnvironment(config = config)

# List of possible base stock values to evaluate, not plotting fractional for convenience in enumeration
base_stock_levels = np.arange(0, config['max_order']+eps, eps)

# data = []
# for tau in base_stock_levels: # loop over each base stock level
#     print(f'Current base stock level: {tau}')
#     base_stock_policy = base_stock.BaseStockPolicy(tau) # set up algorithm

#     for i in range(num_simulations): # simulate the process
#         state = inv_env.reset()

#         for step in range(config['epLen']):
#             action = base_stock_policy.pick_action(state)
#             cost, state, done, info = inv_env.step(action)
#             data_dict = {'iteration' : i, 'step' : step, 'base_stock' : tau, 'cost' : cost}
#             data.append(data_dict)


# df = pd.DataFrame.from_dict(data)
# df.to_csv('./data/base_stock_test.csv')
df = pd.read_csv('./data/base_stock_test.csv')


# Group by 'base_stock' and 'iteration', then sum the rewards
grouped_sum = df.groupby(['base_stock', 'iteration'])['cost'].sum().reset_index()

# Now, average the summed rewards over iterations for each base_stock
result = grouped_sum.groupby('base_stock')['cost'].mean().reset_index()

# Rename the columns for clarity
result.rename(columns={'cost': 'average_cost'}, inplace=True)
# result['average_cost'] = result['average_cost'] / config['epLen']

print(result)
plt.figure(figsize=(10, 7))

# Create the plots and save them
sns.lineplot(data = result, x='base_stock', y='average_cost', color='black', errorbar=None)
plt.xlabel('Base Stock')
plt.ylabel('Average Cost')
# plt.title('Value Function for Base Stock Policies')
# plt.show()
plt.savefig('./figures/base_stock_performance.pdf')

# sns.lineplot()


