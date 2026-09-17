from _de_tuning import run_grid
from _fitness import fitness_fn

if __name__ == '__main__':
    de_results = run_grid(fitness_fn)
    print(f"DE results: {de_results}")