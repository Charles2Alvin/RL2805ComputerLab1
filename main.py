import numpy as np
import maze as mz

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

# Create an environment maze
env = mz.Maze(maze)


# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming
V, policy = mz.dynamic_programming(env, horizon)

print(policy)
method = 'DynProg'
start = (0, 0, 6, 6)
path = env.simulate(start, policy, method)

