import os
import sys
import csv
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from itertools import product

from _de import DE
from _fitness import PARAM_NAMES, BOUNDS, fitness_fn

from agent_42 import StudentAgent 
import test

MAX_ITER = 10
SEED = None


PARAM_GRID = {
    'F' : [0.6, 0.8], # [0.4, 0.6, 0.8, 1.0]
    'CR' : [0.5, 0.7], # [0.5, 0.7, 0.9]
    'population_size' : [30], # [20, 30, 50]
}

def run_grid(fit_fn):
    keys = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))
    print(f"Running {len(combos)} DE configurations ({MAX_ITER} generations each... \n)")

    results = []
    for i, combo, in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        label  = f"F={params['F']}  CR={params['CR']}  pop={params['population_size']}"
        

        de = DE(
            fit_fn, BOUNDS,
            population_size=params['population_size'],
            max_iter=MAX_ITER,
            F=params['F'],
            CR=params['CR'],
            seed=SEED,
        )
        best_theta, best_fit, history = de.optimise()
        
        results.append({
            'label':        label,
            'params':       params,
            'best_fitness': best_fit,
            'best_theta':   best_theta,
            'history':      history,
            'n_evals':      de.n_evals,
             
        })

        print(f"\nRun {i} Completed | {label} | Best Fitness Score: {best_fit:.2f}")
        print("-" * 45)
        print("Optimal Constants Discovered:")

        for name, val in zip(PARAM_NAMES, best_theta):
            if name == 'MAX_D':
                print(f"  {name:<16}: {int(round(val))}")
            else:
                print(f"  {name:<16}: {val:.4f}")
        print("-" * 45 + "\n")
    return results


