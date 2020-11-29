# Xitao Mo 970919-4691
# Jingning Zhou 970413-6705

import maze_rob as mr
import matplotlib.pyplot as plt
import numpy as np
import rob_bank as rb

city = mr.City()
gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
Vs = []

for gamma in gammas:
    print(gamma)
    epsilon = 0.0001
    V, policy = mr.value_iteration(city, gamma, epsilon)
    idx = city.map_[city.start_state]
    Vs.append(V[idx])
    city.draw(policy, plot=True, arrows=True)

plt.plot(gammas, Vs, color='green')
plt.xlabel("Discount factor")
plt.ylabel("Value Function")
plt.show()
path = city.simulate(city.start_state, policy, method="ValIter", survival_factor=gamma)


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

    method = 'DynProg'
    start = (0, 0, 1, 2)

    env = rb.Maze(maze)
    V, policy = rb.dynamic_programming(env)

    env.simulate(start, policy, method)


