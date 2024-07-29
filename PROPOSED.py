import time
import numpy as np

def PROPOSED(population,obj_func, lb,ub, max_generations):
    beta = 0.2
    new_fitness = np.inf
    num_hawks, dim = population.shape[0],population.shape[1]
    convergence_curve = np.zeros((max_generations,num_hawks))
    # Evaluate the objective function for each hawk
    fitness = obj_func(population)
    # Find the best hawk
    best_hawk = population[np.argmin(fitness)]
    ct = time.time()
    # Iterate for a fixed number of generations
    for generation in range(max_generations):
        alpha = np.sqrt(((max_generations *generation )/num_hawks))
        # Update the position of each hawk
        for i in range(num_hawks):
            for j in range(num_hawks):
                if fitness[j] < fitness[i]:
                    r = np.random.uniform()
                    A = 2 * alpha * r - alpha
                    C = 2 * r
                    D = np.abs(C * best_hawk - population[i])
                    new_position = best_hawk - A * D
                    new_position += beta * (np.random.uniform(size=dim) - 0.5)
                    # Check if the new position is within the search space bounds
                    new_position = np.clip(new_position, lb, ub)
                    # Evaluate the objective function for the new position
                    new_fitness = obj_func(new_position)
                    convergence_curve[generation,:] = new_fitness.ravel()
                    # Update the hawk's position and fitness if the new position is better
                    if new_fitness[i] < fitness[i]:
                        population[i] = new_position[i]
                        fitness[i] = new_fitness[i]
        # Find the new best hawk
        new_best_hawk = population[np.argmin(fitness)]
        # Update the best hawk
        if obj_func(new_best_hawk) < obj_func(best_hawk):
            best_hawk = new_best_hawk
        ct = time.time()-ct
    return best_hawk, convergence_curve, new_fitness, ct




