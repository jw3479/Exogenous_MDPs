import numpy as np
from numpy.linalg import matrix_rank
import itertools


"""
Calculating the design matrix for the newsvendor problem with lead time L.
Each row corresponds to the feature vector $phi(s' | s,a)$ which is precisely:
Ind{s' = f(s,a,xi)} as $xi$ enumerates over its support.
"""

def generate_newsvendor_design_matrix(support_size, lead_time = 0):
    state_space_size = support_size ** (lead_time + 1) # state space is [n]^{L+1}
    action_space_size = support_size # action space is [n]
    exogenous_space = support_size # exogenous space is [n]

    state_vectors = list(itertools.product(range(support_size), repeat=lead_time + 1)) # iterator over [n]^{L+1}
    M = np.zeros(((state_space_size ** 2) * (action_space_size), exogenous_space)) # Constructs the design matrix which is S^2 A x d

    index = 0 # index for looping over S^2 A

    for s_new in state_vectors: # loop over new_state
        for a in range(support_size): # loop over a
            for s in state_vectors: # loop over s
                for xi in range(support_size): # loop over xi

                    new_state = np.copy(s_new) # copies the empty matrix (will be completely overwritten)

                    if lead_time == 0: # L = 0 then simple newsvendor dynamics
                        new_state[0] = min(support_size - 1, max( 0 , s[0] + a - xi ))
                            # in the case of no lead-time, on hand inventory rises to s + a, then demand of xi is observed
                    else:
                        new_state[0] = min(support_size - 1, max( 0, s[0] + s[1] - xi))
                            # in the case of no lead-time, on hand inventory rises to s[0] + s[1] (most recent ordered)
                            # then a demnd of xi is observed
                        new_state[1:] = np.append(np.asarray(s[2:]),a)
                            # rest of the state corresponds to the next outgoing orders tacked on with the recent one
                    if np.array_equal(new_state,s_new):
                        M[index, xi] = 1 # Checks if f(s,a,xi) = s_new and sets the corresponding entry to be 1
                index += 1 # updates the index for the next row in the matrix
    return M


"""
Calculating the design matrix for the bin packing problem with bin size B and time horizon T
Each row corresponds to the feature vector $phi(s' | s,a)$ which is precisely:
Ind{s' = f(s,a,xi)} as $xi$ enumerates over its support.
"""


def generate_bin_packing_design_matrix(bin_size, time_horizon):
    state_space_size = (time_horizon ** (bin_size)) * bin_size  # state space is [T]^{B} x [B]
    action_space_size = bin_size # action space is [B]
    exogenous_space = bin_size # exogenous space is [B]

    state_vectors = list(itertools.product(range(time_horizon), repeat=bin_size)) # iterator over [T]^{B+1}

    M = np.zeros(((state_space_size ** 2) * (action_space_size), exogenous_space)) # Constructs the design matrix which is S^2 A x d

    index = 0 # index for looping over S^2 A

    for s_new in state_vectors: # loop over new_state
        for xi_new in range(bin_size): # loops over new_arrival

            for a in range(bin_size): # loop over a

                for s in state_vectors: # loop over s
                    for cur_xi in range(bin_size): # loops over current_arrival

                        for xi in range(bin_size): # loop over xi


                            # print(f'Current state: {s}, current_size: {cur_xi}, action: {a}')
                            new_state = np.copy(s) # copies the empty matrix (will be completely overwritten)

                            if a == 0: # open up a new bin
                                new_state[cur_xi] += 1 # opens up a new bin at current arrival (s[-1])
                                new_state[-1] = xi # current arrival for next timestep is xi

                            elif new_state[a] > 0 and a + cur_xi < bin_size: # checks if action is feasible
                                new_state[a] -= 1 # decreases number of bins at size a
                                new_state[a+cur_xi] += 1 # increments number of bins of size a + current_job_size
                                new_state[-1] = xi # current arrival for next timestep is xi

                            if np.array_equal(new_state,s_new) and xi_new == xi:
                                M[index, xi] = 1 # Checks if f(s,a,xi) = s_new and sets the corresponding entry to be 1
                        index += 1 # updates the index for the next row in the matrix
    return M





print("#########################")
print("#  CHECKING BIN PACKING #")
print("#########################")

for bin_size in np.arange(2,10):
    for time_horizon in np.arange(5,7):
        M = generate_bin_packing_design_matrix(bin_size, time_horizon)
        print(f'Bin size is: {bin_size}, time horizon is: {time_horizon}, and rank of the matrix is: {matrix_rank(M)}')



print("#########################")
print("#  CHECKING NEWSVENDOR  #")
print("#########################")


for support_size in np.arange(1,10):
    for lead_time in np.arange(0,3):
        M = generate_newsvendor_design_matrix(support_size, lead_time)
        # print(f'The design matrix is: \n {M}')
        print(f'Dimension is: {support_size} and rank of the matrix is: {matrix_rank(M)}')