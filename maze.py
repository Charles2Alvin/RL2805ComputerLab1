import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math

# Implemented methods
methods = ['DynProg', 'ValIter', 'PolIter']

# Some colours
RED          = '#FF0000'
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 100
    IMPOSSIBLE_REWARD = -100
    MINOUTAUR_REWARD = -2
    NEAR_MINOUTAUR_REWARD = -2

    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards(weights=weights,
                                                       random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1, 0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] != 1:
                    for m in range(self.maze.shape[0]):
                        for n in range(self.maze.shape[1]):
                            states[s] = (i, j, m, n)
                            map[(i, j, m, n)] = s
                            s += 1
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        i = self.states[state][0]
        j = self.states[state][1]
        m = self.states[state][2]
        n = self.states[state][3]
        
        # Have we arrived at the exit(terminal state) ?
        arrived_exit = (self.maze[i, j] == 2)
        
        # Have we bin eaten by the minotaur(terminal state)
        success = (i == m) and (j == n)
        
        # Is the future position an impossible one ?
        row = i + self.actions[action][0]
        col = j + self.actions[action][1]
        
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row, col] == 1)
        
        # Based on the impossibility check return the next state.
        if hitting_maze_walls or arrived_exit or success:
            return state
        else:
            return self.map[(row, col, m, n)]
        
    def __move_minotaur(self, state):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        
        # Compute the future position given current (state, action)
        i = self.states[state][0]
        j = self.states[state][1]
        m = self.states[state][2]
        n = self.states[state][3]

        # Random action
        action = np.random.randint(1, 5)
        
        # Is the future position an impossible one ?
        row = m + self.actions[action][0]
        col = n + self.actions[action][1]
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) 
        
        # Have we arrived at the exit(terminal state) ?
        success = (self.maze[i, j] == 2)
        
        # Have we bin eaten by the minotaur(terminal state)
        eaten = (i == m) and (j == n)
        
        # Based on the impossibility check return the next state.
        if hitting_maze_walls or success or eaten:
            return state
        else:
            return self.map[(i, j, row, col)]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probabilities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)
                transition_probabilities[next_s, s, a] = 1
        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(0, self.n_actions):
                    next_s = self.__move(s, a)

                    i = self.states[next_s][0]
                    j = self.states[next_s][1]
                    m = self.states[next_s][2]
                    n = self.states[next_s][3]

                    one_step_from_minotaur = (np.abs(i - m) + np.abs(j - n)) == 1

                    if s == next_s and a != self.STAY:
                        # Reward for getting eaten
                        if (i == m) and (j == n):
                            rewards[s, a] = self.MINOUTAUR_REWARD
                            # Reward for reaching the exit
                        elif self.maze[i, j] == 2:
                            rewards[s, a] = self.GOAL_REWARD
                        # Reward for hitting a wall
                        else:
                            rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for getting eaten from minotaur

                    # Reward for reaching the exit
                    elif self.maze[i, j] == 2:
                        rewards[s, a] = self.GOAL_REWARD
                    # Reward for being in the moving range of the minotaur
                    elif one_step_from_minotaur:
                        rewards[s, a] = self.NEAR_MINOUTAUR_REWARD

                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] = self.STEP_REWARD

        return rewards

    def simulate(self, start, policy, method, T=100):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s, t])
                # Add the position in the maze corresponding to the next state after action from policy
                path.append(self.states[next_s])
                # Add the position in the maze corresponding to the next state after random move of minotaur
                next_s = self.__move_minotaur(next_s)
                path.append(self.states[next_s])

                # Check if terminal state
                i = self.states[next_s][0]
                j = self.states[next_s][1]

                # Have we arrived at the exit(terminal state) ?
                success = (self.maze[i, j] == 2)
                if success or t > 2 * T:
                    break
                # Update time and state for next iteration
                t += 1
                s = next_s

        return path

    def repeat_simulate(self, start, policy, method, N, T=100):
        exit_time = []
        for r in range(N):
            path = self.simulate(start, policy, method, T)
            i, j, m, n = path[-1][0], path[-1][1], path[-1][2], path[-1][3]
            if i == m and j == n:
                continue
            elif self.maze[i, j] == 2:
                t = math.ceil(len(path) / 2)
                exit_time.append(t)

        return len(exit_time) / N, exit_time

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

    def draw_optimal_policy(self, maze, minotaur, policy):
        # Map a color to each cell in the maze
        col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

        # Give a color to each cell
        rows, cols = maze.shape

        # Create figure of the size of the maze
        plt.figure(1, figsize=(cols, rows))

        # Remove the axis ticks and add title
        ax = plt.gca()
        ax.set_title('The Maze')
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        rows, cols = maze.shape
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]
        colored_maze[0][0] = LIGHT_ORANGE
        colored_maze[6][5] = LIGHT_RED

        # Create figure of the size of the maze
        plt.figure(1, figsize=(cols, rows))

        # Create a table to color
        grid = plt.table(cellText=None,
                         cellColours=colored_maze,
                         cellLoc='center',
                         loc=(0, 0),
                         edges='closed')
        # Modify the height and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0 / rows)
            cell.set_width(1.0 / cols)

        col_map = {0: '@', 1: '⬅️', 2: '➡️', 3: '⬆️', 4: '⬇️'}
        m = minotaur[0]
        n = minotaur[1]
        for i in range(rows):
            for j in range(cols):
                if maze[i, j] == 0:
                    text = col_map[policy[(self.map[(i, j, m, n)]), 0]]
                    grid.get_celld()[(i, j)].get_text().set_text(text)
        grid.get_celld()[(0, 0)].get_text().set_text("Player")
        grid.get_celld()[(6, 5)].get_text().set_text("Minotaur")


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming backwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows, cols    = maze.shape

    # Create figure of the size of the maze
    plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols    = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    colored_maze[0][0] = LIGHT_ORANGE

    # Create figure of the size of the maze
    plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze, cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
