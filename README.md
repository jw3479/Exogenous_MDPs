# Simulations for "Exploiting Exogenous Structure for Sample-Efficient Reinforcement Learning"

### Folder Overview

1. ``design_matrix`` contains files for calculating the design matrix of the SIS, Inventory Control, and Bin Packing problems
    - ``design_matrix_calculator.py`` computes rank of $F$ for bin packing and inventory control problem
    - ``sis_design_matrix_calculator.py`` computes rank of $F$ for SIS model
    - ``sis_model_svd.ipynb`` computes the low rank feature representation in the SIS model
2. ``inventory_control`` contains files for the numerical simulations conducted in the paper
    - ``base_stock.py`` implements the base stock policy
    - ``base_stock_test.py`` plots $V_1^b$ against base stock level $b$
    - ``check_convergence.py`` is used to test convergence of $Q$ estimates in the UCRL-VTR algorithm
    - ``config_list.py`` lists configuration parameters for the simulation
    - ``find_gap.py`` loops through different configuration parameters to find ones where gap between optimal policy and optimal base stock is large
    - ``helper.py`` implements some helper functions (e.g. binary search for optimal base stock level)
    - ``linear_ucb.py`` implements the UCRL-VTR algorithm with general features
    - ``online_convex_optimization.py`` implements the onlien base stock algorithm
    - ``opt_policy.py`` implements the optimal policy after solving Bellman equations with known $P_x$
    - ``plot_experiment.py`` creates the plots in the paper
    - ``plug_in.py`` implements the plug-in algorithm
    - ``random.py`` implements a randomized policy
    - ``run_experiment.py`` runs the simulation over the different algorithms
    - ``simulator.py`` implements a simulator for a general inventory control problem with lost sales and positive lead time
    - ``tabular_ucb.py`` implements a tabular UCB algorithm

### Experiment Replication

In order to reproduce the experiments conducted in the paper, see the file ``inventory_control/run_experiment.py``.  At the top of the file you can specify:
1. Configuration (i.e. the set-up for the underlying inventory problem with lost sales and positive lead time)
2. Algorithm list
3. Number of simulations

Update the parameters to match the experiment of interest (parameters specified for experiment are in the supplementary material of the paper).  Runnning the file will create a ``.csv`` file of the outputs.  Afterwards, running ``inventory_control/plot_experiment.py`` will generate all of the figures included in the paper.

NOTE: ``environment.yml`` contains all package dependencies.


### References 
1. Reference code from Ayoub et al [(see here)](https://github.com/aa14k/Exploration-in-RL/blob/master/UCRL_VTR.ipynb) used in the ``linear_ucb'' algorithm, but modified to account for user-designed features rather than the natural features from a discrete state-action MDP
2. Learning in structured MDPs with convex cost functions: Improved regret bounds for inventory management by Shipra Agrawal, Randy Jia [(arXiv)](https://arxiv.org/abs/1905.04337)
3. Stochastic convex optimization with bandit feedback by Alekh Agarwal, Dean P. Foster, Daniel Hsu, Sham M. Kakade, Alexander Rakhlin [(arXiv)](https://arxiv.org/abs/1107.1744)

