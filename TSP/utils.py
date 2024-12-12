import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import numpy as np
import json
import random
import threading
import sys
import _thread as thread

from dotenv import load_dotenv
from typing import Literal

load_dotenv()

QISKIT_TOKEN = os.getenv("QISKIT_TOKEN")


def get_defaults(fn):
    if fn.__defaults__ == None:
        return {}
    return dict(zip(
        fn.__code__.co_varnames[-len(fn.__defaults__):],
        fn.__defaults__
    ))


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


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt


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


def dense_distance_matrix(n, max_distance=100):
    """Create a dense distance matrix where all nodes are connected."""
    triangle = np.triu(np.random.randint(
        1, max_distance, size=(n, n)), 1)  # Upper triangle of the matrix
    sym = triangle + triangle.T  # Make the matrix symmetric
    # Set diagonal to zero, as distance to itself is zero
    np.fill_diagonal(sym, 0)
    return sym.tolist()



def __create_dst_mtx():
    with open("problems.json", mode="w") as f:
        problems = {
            "dense": {
                "low": {},
                "high": {}
            },
            "sparse": {
                "low": {},
                "high": {}
            }
        }
        json.dump(problems, f, indent=2)
    return problems


def get_distance_matrices(nodes: list[int] = [4, 5, 6, 7, 8, 10, 11, 12, 15, 20, 69]):
    if not os.path.exists("problems.json"):
        problems = __create_dst_mtx()

    with open("problems.json") as f:
        if os.stat("problems.json").st_size == 0:
            problems = __create_dst_mtx()
        else:
            problems = json.load(f)

    dense = problems["dense"]
    changes = False
    for i in nodes:
        if str(i) not in dense["low"]:
            dense["low"][str(i)] = dense_distance_matrix(i, max_distance=10)
            changes = True
        if str(i) not in dense["high"]:
            dense['high'][str(i)] = dense_distance_matrix(i, max_distance=1000)
            changes = True
            
    if changes:
        with open("problems.json", mode="w") as f:
            json.dump(problems, f, indent=2)

    return problems


if __name__ == "__main__":
    print(dense_distance_matrix(4))
    # p = r"benchmark.csv"
    # data = pd.read_csv(p)
    # print(clean_data(data))
    # with open(p, "w") as f:
    #     data.to_csv(f, index=False)
    # @exit_after(5)
    # def hello():
    #     for i in range(10000000000):
    #         if i % 1000000 == 0:
    #             print(i)

    # hello()
