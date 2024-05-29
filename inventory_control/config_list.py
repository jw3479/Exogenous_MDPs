import numpy as np
import scipy


def get_small_config(mean, c=6, h=4):

    distr = [scipy.stats.poisson.pmf(a, mu=mean) for a in range(3)]
    config = {
          'name': 'small_config_'+str(mean),
          'lead_time': 2,
          'sales_penalty': c,
          'holding_cost' : h,
          'epLen' : 10,
          'demand_dist' : distr / np.sum(distr),
          'max_order' : 2,
          'max_inventory' : 2, 
          'starting_state': np.asarray([0.,0.,0.])
          }

    return config


def get_medium_config(mean, c=6, h=4):

    distr = [scipy.stats.poisson.pmf(a, mu=mean) for a in range(4)]
    print(f'Mean: {np.dot(np.arange(0, 4), distr / np.sum(distr))}')
    print(f'Distribution: {distr / np.sum(distr)}')
    config = {
          'name': 'medium_config_'+str(mean)+'_'+str(c)+'_'+str(h),
          'lead_time': 1,
          'sales_penalty': c,
          'holding_cost' : h,
          'epLen' : 15,
          'demand_dist' : distr / np.sum(distr),
          'max_order' : 3,
          'max_inventory' : 3, 
          'starting_state': np.asarray([0.,0.])
          }

    return config


def get_no_lead_config(mean, c=6, h=4, support=10):

    distr = [scipy.stats.poisson.pmf(a, mu=mean) for a in range(support+1)]
    print(f'Distribution: {distr / np.sum(distr)}')
    config = {
          'name': 'no_lead_config_'+str(mean)+'_'+str(c)+'_'+str(h),
          'lead_time': 0,
          'sales_penalty': c,
          'holding_cost' : h,
          'epLen' : 20,
          'demand_dist' : distr / np.sum(distr),
          'max_order' : support,
          'max_inventory' : support, 
          'starting_state': np.asarray([0.])
          }

    return config
