import numpy as np
from numpy.linalg import matrix_rank
import itertools


"""
Calculating the design matrix for the SIS model with n individuals.
"""

def generate_sis_design_matrix(n = 1):
    # Original SIS Model: Each timestep, all people transition either infect / uninfected
    # Action: Vaccine decisions for EACH individual (i.e. unrestricted budget)
    state_space_size = 2 ** (n) # state space is {0,1}^n
    action_space_size = 2 ** (n) # action space is {0,1}^n
    exogenous_space = 2 ** (3 * n) # exogenous space is {0,1}^3n

    M = np.zeros(((state_space_size ** 2) * (action_space_size), exogenous_space)) # Constructs the design matrix which is S^2 A x d

    index = 0 # index for looping over S^2 A

    for s in itertools.product([0,1], repeat=n): # loop over (s,a,s')
        for a in itertools.product([0,1], repeat=n):
            for s_new in itertools.product([0,1], repeat=n):
                xi_index = 0 # index looping over \Xi
                for xi in itertools.product([0,1], repeat=3*n):
                    ar_xi = np.asarray(xi)
                    new_state = np.asarray(np.copy(s))
                    for i in range(n):
                        # print(f'person: {i}')
                        if s[i] == 0 and a[i] == 0:
                            new_state[i] = ar_xi[3*i]
                        elif s[i] == 0 and a[i] == 1:
                            new_state[i] = ar_xi[3*i+1]
                        elif s[i] == 1:
                            new_state[i] = ar_xi[3*i+2]
                    if np.array_equal(new_state, np.asarray(s_new)):
                        M[index, xi_index] = 1
                    xi_index += 1
                index += 1
    return M


def generate_sis_design_matrix_single_action(n = 1):
    # Original SIS Model: Each timestep, all people transition either infect / uninfected
    # Action: Single vaccine allocation / no vaccine administered
    state_space_size = 2 ** (n) # state space is {0,1}^n
    action_space_size = n+1 # action space is {0,1}^n
    exogenous_space = 2 ** (3 * n) # exogenous space is {0,1}^3n

    M = np.zeros(((state_space_size ** 2) * (action_space_size), exogenous_space)) # Constructs the design matrix which is S^2 A x d

    index = 0 # index for looping over S^2 A

    for s in itertools.product([0,1], repeat=n): # loop over (s,a,s')
        # for a in itertools.product([0,1], repeat=n):
        for a in range(n+1):
            for s_new in itertools.product([0,1], repeat=n):
                xi_index = 0 # index looping over \Xi
                for xi in itertools.product([0,1], repeat=3*n):
                    ar_xi = np.asarray(xi)
                    new_state = np.asarray(np.copy(s))
                    for i in range(n):
                        # print(f'person: {i}')
                        if s[i] == 0 and a == i:
                            new_state[i] = ar_xi[3*i]
                        elif s[i] == 0 and a == i:
                            new_state[i] = ar_xi[3*i+1]
                        elif s[i] == 1:
                            new_state[i] = ar_xi[3*i+2]
                    if np.array_equal(new_state, np.asarray(s_new)):
                        M[index, xi_index] = 1
                    xi_index += 1
                index += 1
    return M




def generate_sis_design_matrix_joint(n = 1):
    # SIS Design Matrix when all individuals share latent node failures, i.e.
    # a single \xi = (\xi_0, \xi_1, \xi_2) is drawn which dictates the infections for ALL individuals

    state_space_size = 2 ** (n) # state space is [n]^{L+1}
    action_space_size = 2 ** (n) # action space is [n]
    exogenous_space = 2 ** (3) # exogenous space is [n]

    M = np.zeros(((state_space_size ** 2) * (action_space_size), exogenous_space)) # Constructs the design matrix which is S^2 A x d

    index = 0 # index for looping over S^2 A

    for s in itertools.product([0,1], repeat=n): # loop over new_state
        for a in itertools.product([0,1], repeat=n):
            for s_new in itertools.product([0,1], repeat=n):
                xi_index = 0
                for xi in itertools.product([0,1], repeat=3):
                    ar_xi = np.asarray(xi)
                    # print(ar_xi)
                    new_state = np.asarray(np.copy(s))
                    for i in range(n):
                        # print(f'person: {i}')
                        if s[i] == 0 and a[i] == 0:
                            new_state[i] = ar_xi[0]
                        elif s[i] == 0 and a[i] == 1:
                            new_state[i] = ar_xi[1]
                        elif s[i] == 1:
                            new_state[i] = ar_xi[2]
                    if np.array_equal(new_state, np.asarray(s_new)):
                        M[index, xi_index] = 1
                    xi_index += 1
                index += 1
    return M



print("#########################")
print("#  CHECKING SIS ORIGINAL #")
print("#########################")

for n in np.arange(1,4):
# for n in [2]:
# for n in [2]:
    M = generate_sis_design_matrix(n)
    # print(M)
    print(M.shape)
    print(f'Number of individuals is: {n}, and rank of the matrix is: {matrix_rank(M)}')


print("#########################")
print("#  CHECKING SIS SINGLE ALLOCATION #")
print("#########################")

for n in np.arange(1,4):
# for n in [2]:
# for n in [2]:
    M = generate_sis_design_matrix_single_action(n)
    # print(M)
    print(M.shape)
    print(f'Number of individuals is: {n}, and rank of the matrix is: {matrix_rank(M)}')




print("#########################")
print("#  CHECKING SIS SINGLE RANDOM VARIABLE #")
print("#########################")

for n in np.arange(1,4):
# for n in [2]:
# for n in [2]:
    M = generate_sis_design_matrix_joint(n)
    # print(M)
    print(M.shape)
    print(f'Number of individuals is: {n}, and rank of the matrix is: {matrix_rank(M)}')


