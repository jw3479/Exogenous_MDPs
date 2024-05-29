import numpy as np
import pandas as pd
import base_stock
import simulator



def export_legend(legend, filename="LABEL_ONLY.pdf"):
    """
        Saves a PDF of only the legend from a given plot
    """
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)




def evaluate_base_stock(inv_env, num_simulations, base_stock_level):
    '''
        Function which estimates the average cost for a base stock policy
        through Monte Carlo simulation
    '''

    base_stock_policy = base_stock.BaseStockPolicy(base_stock_level) # creates the base stock policy

    tot_cost = 0
    for i in range(num_simulations):
        state = inv_env.reset()

        for step in range(inv_env.config['epLen']):
            action = base_stock_policy.pick_action(state)
            cost, state, done, info = inv_env.step(action) # evaluates the cost and average
            
            tot_cost += cost
    return tot_cost / (num_simulations)




def get_optimal_base_stock(inv_env, num_simulations, low = 10, high = 20, tol=1, INT_FLAG=True):
    '''
    Function to get the optimal base stock value. Takes as input an inventory environment and base stock range
    and performs bisection search in order to converge to the optimal.

    Args:
        - inv_env: Simulator for environment
        - num_simulations: number of simulations
        - low: low value for base stock to check
        - high: high value for base stock to check
        - tol: error tolerance
        - INT_FLAG: flag to only search and check integral base stock values

    Returns:
        - optimal base stock value
        - optimal base stock cost
    '''

    if INT_FLAG:
        tau_low = int(low)
        tau_high = int(high)
    else:
        tau_low = low
        tau_high = high


    while tau_high - tau_low > tol: # tolerance for converging to optimal base stock value
        print(f'Current Interval: {tau_low, tau_high}')
        # Choose two points in the middle third and last third
        m1 = tau_low + (tau_high - tau_low) / 3
        m2 = tau_high - (tau_high - tau_low) / 3

        if INT_FLAG and tau_high - tau_low > 2:
            m1 = int(m1)
            m2 = int(m2)

        elif INT_FLAG and tau_high - tau_low <= 2: # just check the three points at this point
            f_1 = evaluate_base_stock(inv_env, num_simulations, tau_low)
            f_2 = evaluate_base_stock(inv_env, num_simulations, tau_low+1)
            f_3 = evaluate_base_stock(inv_env, num_simulations, tau_high)
            if f_1 <= min(f_2, f_3):
                return tau_low, f_1
            elif f_2 <= min(f_1, f_3):
                return tau_low+1, f_2
            else:
                return tau_high, f_3

        # Evaluate the performance at these base stock values
        f_m1 = evaluate_base_stock(inv_env, num_simulations, m1)
        f_m2 = evaluate_base_stock(inv_env, num_simulations, m2)
        
        # Narrow down the search interval based on where the minimum is likely to be
        if f_m1 < f_m2:
            tau_high = m2  # The minimum is in the left segment
        else:
            tau_low = m1  # The minimum is in the right segment

    tau_min = (tau_low + tau_high) / 2
    obj_value = evaluate_base_stock(inv_env, num_simulations, tau_min) # call recursively

    return tau_min, obj_value