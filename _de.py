import numpy as np
import multiprocessing
import concurrent.futures

class DE: 
    """
    Differential Evolution (DE) - 
    Storn, R., Price, K. Differential Evolution – 
    A Simple and Efficient Heuristic for global Optimization over Continuous Spaces. 
    Journal of Global Optimization 11, 341–359 (1997). 
    https://doi.org/10.1023/A:1008202821328
    

    Parameters to be tweaked:
    - fitness_fn        : callable(theta), being maximised
    - bounds            : state space
    - population_size   : number of candidate solutions
    - max_iter          : number of generations
    - F                 : mutation factor [values]
    - CR                : crossover rate [values]
    - seed              : for reproduceability 
    """

    def __init__(self, fitness_fn, bounds, population_size, max_iter, F, CR, seed=None):
        self.fitness_fn = fitness_fn
        self.bounds = np.array(bounds, dtype=float)
        self.population_size = population_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.rng = np.random.default_rng(seed)

    def optimise(self):
        """
        Runs DE and returns the best solution found
        """
        lb, ub, n_dim = self.bounds[:, 0], self.bounds[:,1], len(self.bounds)
        population = self.rng.uniform(lb, ub, size=(self.population_size, n_dim))

        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness[i] = self.fitness_fn(population[i])

        best_index = np.argmax(fitness)
        best_theta = population[best_index].copy()
        best_fitness = fitness[best_index]
        history = [best_fitness]

        for _ in range(self.max_iter):
            indices = np.zeros((self.population_size, 3), dtype=int)
            for i in range(self.population_size):
                while True:
                    abc = self.rng.choice(self.population_size, size=3, replace=False)
                    if i not in abc:
                        indices[i] = abc
                        break
                        
            a, b, c = indices[:, 0], indices[:, 1], indices[:, 2]

            mutants = np.clip(population[a] + self.F * (population[b] - population[c]), lb, ub)
            
            crossover_masks = self.rng.random((self.population_size, n_dim)) < self.CR
            j_rands = self.rng.integers(0, n_dim, size=self.population_size)
            crossover_masks[np.arange(self.population_size), j_rands] = True

            trials = np.where(crossover_masks, mutants, population)

            safe_workers = max(1, multiprocessing.cpu_count() - 1)

            with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
                trial_fitnesses = np.array(list(executor.map(self.fitness_fn, trials)))

            better_mask = trial_fitnesses >= fitness
            
            population[better_mask] = trials[better_mask]
            fitness[better_mask] = trial_fitnesses[better_mask]

            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_fitness:
                best_fitness = fitness[best_idx]
                best_theta = population[best_idx].copy()

            history.append(best_fitness)
        return best_theta, best_fitness, history


    @property
    def n_evals(self):
        return self.population_size * (self.max_iter + 1)