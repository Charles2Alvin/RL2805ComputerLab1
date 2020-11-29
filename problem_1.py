# Xitao Mo 970919-4691
# Jingning Zhou 970413-6705

import numpy as np
import maze as mz
from matplotlib import pyplot as plt


def run():
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])
    # with the convention
    # 0 = empty cell
    # 1 = obstacle
    # 2 = exit of the Maze

    method = 'DynProg'
    start = (0, 0, 6, 6)
    N = 100

    exit_prob = []

    horizons = [i for i in range(20, 41)]

    for horizon in horizons:
        # Create an environment maze
        env = mz.Maze(maze)

        # Solve the MDP problem with dynamic programming
        V, policy = mz.dynamic_programming(env, horizon)

        # Repeat the simulation for N times
        prob, exit_time = env.repeat_simulate(start, policy, method, N)

        exit_prob.append(prob)

        print("Completed simulation for T = %s, obtained exit prob = %.3f" % (horizon, prob))

    plt.plot(horizons, exit_prob)
    plt.xlabel("Time horizon")
    plt.ylabel("Escape probability")
    plt.show()


def random_horizon():
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    method = 'DynProg'
    start = (0, 0, 6, 6)
    N = 10000

    # Create an environment maze
    env = mz.Maze(maze)

    # Solve the MDP problem with dynamic programming
    V, policy = mz.dynamic_programming(env, 18)

    # Sample the horizon from the distribution
    horizon = int(np.random.normal(30, 1))

    prob, exit_time = env.repeat_simulate(start, policy, method, horizon, N)

    mean = np.average(exit_time)

    print("Expected time to exit = %.4f, success prob = %.4f" % (mean, len(exit_time)/N))




