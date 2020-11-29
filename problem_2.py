# Xitao Mo 970919-4691
# Jingning Zhou 970413-6705

import matplotlib.pyplot as plt
import numpy as np
import rob_bank as rb


gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
Vs = []


def run():
    # Description of the maze as a numpy array
    maze = np.array([
        [1, 0, 0, 0, 0, 1],
        [0, 0, 2, 0, 0, 0],
        [1, 0, 0, 0, 0, 1]
    ])
    # with the convention
    # 0 = empty cell
    # 1 = bank
    # 2 = police station

    env = rb.Maze(maze)
    gamma = 0.4
    epsilon = 0.0001
    V, policy = rb.value_iteration(env, gamma, epsilon)

    method = 'ValIter'
    start = (0, 0, 1, 2)
    path = env.simulate(start, policy, method)

    env.simulate(start, policy, method)

    rb.animate_solution(maze, path)

    env.draw_optimal_policy(maze, policy)


