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

plt.style.use('PaperDoubleFig.mplstyle.txt')

# File to plot the different algorithms performance

INCLUDE_BASE_OPT = True # indicator for whether to include dotted line for optimal
INCLUDE_TRUE_OPT = True
plot_frequency = 50

seed = 5

# Config parameters
# config = config_list.get_medium_config(3, 6, 1)
config = config_list.get_no_lead_config(7,6,4)

# ALGORITHMS FOR SCENARIO I
# algo_list = ['true_plug_in', 'online_base_stock_1_100', 'linear_ucb_1']

# ALGORITHMS FOR SCENARIO II
algo_list = ['true_plug_in', 'linear_ucb_1_10', 'online_base_stock_1_100']


# name_mapping = {
#     'true_plug_in' : 'Plug In',
#     'linear_ucb_1_10' : 'UCRL',
#     'online_base_stock_1_100': 'Online Base Stock'
# }


colors = [
    (164/255, 108/255, 183/255),  # Converted RGB for the first algorithm
    (122/255, 164/255, 87/255),   # Converted RGB for the second algorithm
    (203/255, 106/255, 73/255)    # Converted RGB for the third algorithm
]



# Setting up the inventory environment
inv_env = simulator.InventoryEnvironment(config = config)


if INCLUDE_BASE_OPT:
    # Calculate optimal base stock policy as a form of regret metric
    opt_base_stock, min_base_cost = helper.get_optimal_base_stock(inv_env, int(1e4), low = 0, high = config['max_order'])
    print(f"Optimal Base Stock Level: {opt_base_stock} and its cost: {min_base_cost}")

if INCLUDE_TRUE_OPT:
    opt_algo = opt_policy.Optimal_Policy(inv_env, config['max_order']+1)
    opt_value = opt_algo.V[0, inv_env.get_key(inv_env.starting_state)]
    opt_value = opt_value
    print(f'Optimal Cost: {opt_value}')

# PARAMETERS FOR SCENARIO I

# min_base_cost = 3.2550761904719034
# opt_value = 1.6745267147387575

# PARAMETERS FOR SCENARIO II

# min_base_cost = 1.655084000002587
# opt_value = 1.655084000002587



df = pd.DataFrame()
# Loop over the algorithms in the list, reading in their dataframes into a giant dataframe to combine
for algo_name in algo_list:
    log_file = './data/inv_data_'+str(seed)+'_'+str(algo_name)+'_'+config['name']+'.csv'
    algo_df = pd.read_csv(log_file) # reads in the dataframe
    df = pd.concat([df, algo_df], ignore_index=True)



grouped_df = df.groupby(['algorithm', 'iteration', 'episode'])['cost'].sum().reset_index() # groups by and sums up the cumulative cost
grouped_df['cost'] = grouped_df['cost'] * config['epLen']


# Create a custom palette based on the three colors
palette = sns.color_palette(colors)


fig, ax = plt.subplots(figsize=(10, 10))  # You can specify the figure size here

spliced_df = grouped_df[grouped_df['episode'] % plot_frequency == 0]
# sns.lineplot(data = spliced_df, x = 'episode', y = 'cost', hue = 'algorithm', palette=palette, style = 'algorithm') # plots algorithm performance per episode
sns.lineplot(data = spliced_df, x = 'episode', y = 'cost', hue = 'algorithm', palette=palette, style = 'algorithm', legend=None) # plots algorithm performance per episode


if INCLUDE_BASE_OPT: plt.axhline(y=min_base_cost*config['epLen'], color='grey', linestyle='-', label='Optimal Base Stock') # add line for optimal base stock value
if INCLUDE_TRUE_OPT: plt.axhline(y=opt_value*config['epLen'], color='black', linestyle='-', label='Optimal Policy')
# plt.legend(title='Legend')

plt.xlabel('Episode')
plt.ylabel('Performance')

plt.savefig('./figures/plot_'+str(seed)+'_'+config['name']+'.pdf')



# legend = ax.legend(ncol = 6, loc= 'lower center', bbox_to_anchor=(-1, -.3, 0.5, 0.5))
# print(len(legend.get_lines()))
# [legend.get_lines()[i].set_linewidth(3) for i in range(len(legend.get_lines()))]

# helper.export_legend(legend, filename="./figures.legend.pdf")


