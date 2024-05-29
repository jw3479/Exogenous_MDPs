import numpy as np
import sys



class BaseStockPolicy():
    """
    Algorithm for a base stock policy heuristic in an inventory control model with positive lead time and lost sales

    Attributes:
        base_stock: the case stock level
    """

    def __init__(self, base_stock):
        self.base_stock = base_stock

    def get_base_stock(self):
        return self.base_stock
    
    def pick_action(self, state):
        """
        Picks an action according to the base stock level.

        Args:
            state: current inventory orders in the system

        Returns:
            action: an ordering amount
        
        """

        total_inventory = np.sum(state) # calculates the total inventory in the pipeline

        if total_inventory <= self.base_stock: # if the total inventory is smaller than the base stock level
            return (self.base_stock - total_inventory) # order the difference

        else:
            return 0 # otherwise order nothing


