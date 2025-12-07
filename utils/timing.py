import time
from functools import wraps
from ..Algorithm.bat_algo_max_flow import bat_max_flow
from ..Algorithm.ga_algo_max_flow import CompleteMaxFlowGA

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMEIT] Function '{func.__name__}' finished in {end-start:.4f} s")
        return result
    return wrapper

@timeit
def run_bat_with_timing(df,source,sink):
    return bat_max_flow(
        df, source, sink,
        max_iterations=300,
        n_bats=5,
        verbose=True,
        drawing=False
    )

@timeit
def run_ga_with_timing(df,source,sink):
    ga=CompleteMaxFlowGA(verbose=False)
    return ga.run_max_flow_ga(df, source, sink,
                        population_size=50,
                        max_generations=400,
                        mutation_rate=0.2,
                        crossover_rate=0.8,
                        verbose=False
    )
