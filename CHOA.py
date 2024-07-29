import time
from math import inf
import numpy as np
from numpy import pi

def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = ub.shape[2 - 1]
    # If the boundaries of all variables are equal and user enter a signle
    # number for both ub and lb
    if Boundary_no == 1:
        Positions = np.multiply(np.random.rand(SearchAgents_no, dim), (ub - lb)) + lb
    # If each variable has a different lb and ub
    if Boundary_no > 1:
        for i in np.arange(1, dim + 1).reshape(-1):
            ub_i = ub(i)
            lb_i = lb(i)
            Positions[:, i] = np.multiply(np.random.rand(SearchAgents_no, 1), (ub_i - lb_i)) + lb_i
    return Positions

def chaos(index, max_iter, Value):
    O = np.zeros((1, max_iter))
    x = 0.7
    if 1 == index:
        for i in np.arange(1, max_iter + 1).reshape(-1):
            x = np.cos(i * np.arccos(x[i]))
            G = ((x[i] + 1) * Value) / 2
    else:
        if 2 == index:
            # Circle map
            a = 0.5
            b = 0.2
            for i in np.arange(1, max_iter + 1).reshape(-1):
                x = np.mod(x[i] + b - (a / (2 * pi)) * np.sin(2 * pi * x[i]), 1)
                G= x(i) * Value
        else:
            if 3 == index:
                # Gauss/mouse map
                for i in np.arange(1, max_iter + 1).reshape(-1):
                    if x== 0:
                        x= 0
                    else:
                        x= np.mod(1 / x[i], 1)
                    G = x * Value
            else:
                if 4 == index:
                    # Iterative map
                    a = 0.7
                    for i in np.arange(1, max_iter + 1).reshape(-1):
                        x = np.sin((a * pi) / x[i])
                        G= ((x + 1) * Value) / 2
                else:
                    if 5 == index:
                        # Logistic map
                        a = 4
                        for i in np.arange(max_iter ):
                            x = a * x[i] * (1 - x[i])
                            G= x(i) * Value
                    else:
                        if 6 == index:
                            # Piecewise map
                            P = 0.4
                            for i in np.arange(1, max_iter + 1).reshape(-1):
                                if x >= 0 and x< P:
                                    x = x/ P
                                if x >= P and x< 0.5:
                                    x[i + 1] = (x- P) / (0.5 - P)
                                if x >= 0.5 and x < 1 - P:
                                    x[i + 1] = (1 - P - x) / (0.5 - P)
                                if x >= 1 - P and x < 1:
                                    x[i + 1] = (1 - x) / P
                                G = x * Value
                        else:
                            if 7 == index:
                                # Sine map
                                for i in np.arange(1, max_iter + 1).reshape(-1):
                                    x = np.sin(pi * x)
                                    G = (x(i)) * Value
                            else:
                                if 8 == index:
                                    # Singer map
                                    u = 1.07
                                    for i in np.arange(1, max_iter + 1).reshape(-1):
                                        x = u * (7.86 * x- 23.31 * (x ** 2) + 28.75 * (
                                                    x ** 3) - 13.302875 * (x ** 4))
                                        G= (x(i)) * Value
                                else:
                                    if 9 == index:
                                        # Sinusoidal map
                                        for i in np.arange(1, max_iter + 1).reshape(-1):
                                            x = 2.3 * x ** 2 * np.sin(pi * x)
                                            G = (x(i)) * Value
                                    else:
                                        if 10 == index:
                                            # Tent map
                                            x = 0.6
                                            for i in np.arange(1, max_iter + 1).reshape(-1):
                                                if x < 0.7:
                                                    x = x/ 0.7
                                                if x >= 0.7:
                                                    x= (10 / 3) * (1 - x)
                                                G = (x) * Value
                                    G = O


    return O

def CHOA(Positions,fobj, lb, ub, Max_iter ):
    # initialize Attacker, Barrier, Chaser, and Driver
    dim = Positions.shape[1]
    Attacker_pos = np.zeros((1, dim))
    Attacker_score = inf
    Barrier_pos = np.zeros((1, dim))
    Barrier_score = inf
    Chaser_pos = np.zeros((1, dim))
    Chaser_score = inf
    Driver_pos = np.zeros((1, dim))
    Driver_score = inf
    # Initialize the positions of search agents
    Convergence_curve = np.zeros((1, Max_iter))
    l = 0
    ##
    # Main loop
    ct = time.time()
    while l < Max_iter:
        for i in np.arange(1, Positions.shape[1 - 1] + 1).reshape(-1):
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i,:] > ub
            Flag4lb = Positions[i,:] < lb
            Positions[i, :] = (np.multiply(Positions[i,:], (not (Flag4ub + Flag4lb)))) + np.multiply(ub, Flag4ub) + np.multiply(lb, Flag4lb)
            # Calculate objective function for each search agent
            fitness = fobj(Positions[i,:])
            # Update Attacker, Barrier, Chaser, and Driver
            if fitness < Attacker_score:
                Attacker_score = fitness
                Attacker_pos = Positions[i,:]
                if fitness > Attacker_score and fitness < Barrier_score:
                    Barrier_score = fitness
                    Barrier_pos = Positions[i,:]
                    if fitness > Attacker_score and fitness > Barrier_score and fitness < Chaser_score:
                        Chaser_score = fitness
                        Chaser_pos = Positions[i,:]
                        if fitness > Attacker_score and fitness > Barrier_score and fitness > Chaser_score and fitness > Driver_score:
                            Driver_score = fitness
                            Driver_pos = Positions[i,:]
                            f = 2 - l * ((2) / Max_iter)
                            #  The Dynamic Coefficient of f Vector as Table 1.
                            # Group 1
                            C1G1 = 1.95 - ((2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)))
                            C2G1 = (2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)) + 0.5
                            # Group 2
                            C1G2 = 1.95 - ((2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)))
                            C2G2 = (2 * (l ** 3) / (Max_iter ** 3)) + 0.5
                            # Group 3
                            C1G3 = (- 2 * (l ** 3) / (Max_iter ** 3)) + 2.5
                            C2G3 = (2 * l ** (1 / 3)) / (Max_iter ** (1 / 3)) + 0.5
                            # Group 4
                            C1G4 = (- 2 * (l ** 3) / (Max_iter ** 3)) + 2.5
                            C2G4 = (2 * (l ** 3) / (Max_iter ** 3)) + 0.5
                            # Update the Position of search agents including omegas
                            for i in np.arange(1, Positions.shape[1 - 1] + 1).reshape(-1):
                                for j in np.arange(1, Positions.shape[2 - 1] + 1).reshape(-1):
                                    ## Please note that to choose a other groups you should use the related group strategies
                                    r11 = C1G1 * np.random.rand()
                                    r12 = C2G1 * np.random.rand()
                                    r21 = C1G2 * np.random.rand()
                                    r22 = C2G2 * np.random.rand()
                                    r31 = C1G3 * np.random.rand()
                                    r32 = C2G3 * np.random.rand()
                                    r41 = C1G4 * np.random.rand()
                                    r42 = C2G4 * np.random.rand()
                                    A1 = 2 * f * r11 - f
                                    C1 = 2 * r12
                                    ## # Please note that to choose various Chaotic maps you should use the related Chaotic maps strategies
                                    m = chaos(3, 1, 1)
                                    D_Attacker = np.abs(C1 * Attacker_pos(j) - m * Positions(i, j))
                                    X1 = Attacker_pos(j) - A1 * D_Attacker
                                    A2 = 2 * f * r21 - f
                                    C2 = 2 * r22
                                    D_Barrier = np.abs(C2 * Barrier_pos(j) - m * Positions(i, j))
                                    X2 = Barrier_pos(j) - A2 * D_Barrier
                                    A3 = 2 * f * r31 - f
                                    C3 = 2 * r32
                                    D_Driver = np.abs(C3 * Chaser_pos(j) - m * Positions(i, j))
                                    X3 = Chaser_pos(j) - A3 * D_Driver
                                    A4 = 2 * f * r41 - f
                                    C4 = 2 * r42
                                    D_Driver = np.abs(C4 * Driver_pos(j) - m * Positions(i, j))
                                    X4 = Chaser_pos(j) - A4 * D_Driver
                                    Positions[i, j] = (X1 + X2 + X3 + X4) / 4
                            l = l + 1
        Convergence_curve[l] = Attacker_score
        ct = time.time()-ct
    return Attacker_score, Attacker_pos, Convergence_curve,ct