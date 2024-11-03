import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import time
import pandas as pd
import numpy as np
import datetime
import threading
import sys
import _thread as thread

from dotenv import load_dotenv


load_dotenv()

QISKIT_TOKEN = os.getenv("QISKIT_TOKEN")


def get_defaults(fn):
    if fn.__defaults__ == None:
        return {}
    return dict(zip(
        fn.__code__.co_varnames[-len(fn.__defaults__):],
        fn.__defaults__
    ))


def benchmark(path="benchmark.csv"):
    # Create the file if it does not exist
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Write header only if the file does not exist or is empty
            writer.writerow(["number_of_cities",
                             "solve_cost", "execution_time", "tour", "solver"])

    def decoration(func):
        def wrapper(*args, **kwargs):
            # Retrieve the argument names to identify `solver` and other arguments
            func_args = func.__code__.co_varnames

            solver = kwargs.get("solver")
            if not solver:
                try:
                    solver_index = func_args.index("solver")
                    if solver_index < len(args):
                        solver = args[solver_index]
                    else:
                        solver = func.__defaults__[solver_index - len(args)]
                except ValueError:
                    pass

            dur, cost, reordered = func(*args, **kwargs)
            # Append the results to the file
            add_data(len(reordered), cost, dur,
                     reordered, solver or func.__name__)

        return wrapper
    return decoration


def calculate_time(func):
    def inner1(*args, **kwargs):

        # storing time before function execution
        begin = time.time()
        
        val = func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        print("Total time taken in : ", func.__name__, end - begin)
        return val, end - begin

    return inner1

def add_data(number_of_cities, solve_cost, execution_time, tour, solver, path="benchmark.csv"):
    with open(path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([number_of_cities, solve_cost, execution_time,
                        tour, solver])

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt
    
def exit_after(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer

def clean_data(data):
    # Function to convert solve_cost to seconds (float) for both formats
    def convert_solve_cost(value):
        if isinstance(value, str):
            if ":" not in value:
                sec, mili = value.split(".")
                hour = int(sec) // 3600
                sec = int(sec) % 3600
                minute = sec // 60
                sec = sec % 60

                return f"{hour:01d}:{minute:02d}:{sec:02d}.{mili[:6].ljust(6, '0')}"
            else:
                return value
        else:
            return value  # Already in seconds as float

    # Function to convert tour into a tuple
    def convert_tour(value):
        # If the tour is already a string (tuple representation), evaluate it
        if isinstance(value, str):
            try:
                return tuple(map(int, eval(value)))
            except:
                return tuple(map(int, eval(value.replace(" ", ","))))  # If evaluation fails, keep the original
        elif isinstance(value, np.ndarray):
            # Convert numpy array to tuple
            return tuple(value)
        else:
            return value  # Already in the correct format

    # Apply conversion functions to relevant columns
    data['execution_time'] = data['execution_time'].apply(convert_solve_cost)
    data['tour'] = data['tour'].apply(convert_tour)

    return data

if __name__ == "__main__":
    p = r"benchmark.csv"
    data = pd.read_csv(p)
    print(clean_data(data))
    with open(p, "w") as f:
        data.to_csv(f, index=False)
    # @exit_after(5)
    # def hello():
    #     for i in range(10000000000):
    #         if i % 1000000 == 0:
    #             print(i)
                
    # hello()