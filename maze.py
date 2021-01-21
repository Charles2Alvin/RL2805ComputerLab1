import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math
import random
import os


"""
Grader: Alessio. Fail. Please, resubmit. 

You need to address the comments regarding P1 (Problem 1). 

Comments: (P1) Observe that there is no need to have a fancy reward function. 
Just assigning a reward of 1 whenever the agent leaves the maze (and 0 otherwise) is 
a way to maximize the probability of leaving the maze. 

It is not clear if you allow the minotaur to walk inside the walls 
(this is an important assumption). 

The results for problem 1.b are wrong 
(observe that the minimum number of steps required to leave the maze is 15, 
and there should be a good probability for the player to leave the maze in 15 steps...
from your plot the player has 20% probability after 20 steps!) 

In problem 1.c what you do is not clear at all. 
What is the probability of leaving the maze? How do you compute the policy? 
Do you use a discount factor? 
How do you take into consideration the life being geometrically distributed?
"""
# Implemented methods
methods = ['DynProg', 'ValIter', 'PolIter']

# Some colours
RED = '#FF0000'
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = -100
    MINOUTAUR_REWARD = -1

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)

        return actions

    def __states(self):
        """
        states: map integers to 4-tuples/WIN/DEAD
        map: map 4-tuples/WIN/DEAD to integers
        """
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
        states[s] = 'WIN'
        map['WIN'] = s
        s += 1
        states[s] = 'DEAD'
        map['DEAD'] = s
        s += 1

        return states, map

    def is_win(self, s):
        if self.states[s] == 'WIN':
            return True
        if self.states[s] == 'DEAD':
            return False
        px, py, mx, my = self.states[s]

        return self.maze[px, py] == 2 and not self.is_dead(s)

    def is_dead(self, s):
        if self.states[s] == 'DEAD':
            return True
        if self.states[s] == 'WIN':
            return False
        px, py, mx, my = self.states[s]

        return px == mx and py == my

    def is_hitting_wall(self, row, col):
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1)

        return hitting_maze_walls

    def __move(self, s, a):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        if self.states[s] == 'WIN' or self.is_win(s):
            return self.map['WIN']
        if self.states[s] == 'DEAD' or self.is_dead(s):
            return self.map['DEAD']

        # Compute the future position given current (state, action)
        row = self.states[s][0] + self.actions[a][0]
        col = self.states[s][1] + self.actions[a][1]

        # Based on the impossibility check return the next state.
        if self.is_hitting_wall(row, col):
            return s
        else:
            m = self.states[s][2]
            n = self.states[s][3]
            # move the minotaur
            possible_moves = self.possible_minotaur_moves(s)
            random_move = random.choice(possible_moves)
            m += random_move[0]
            n += random_move[1]

            return self.map[(row, col, m, n)]

    def possible_minotaur_moves(self, s):
        # Get the (x, y) coordinates of the minotaur
        mx, my = self.states[s][2], self.states[s][3]

        possible_moves = []
        for i in range(self.n_actions):
            dx = self.actions[i][0]
            dy = self.actions[i][1]
            row = mx + dx
            col = my + dy

            if (row == -1) or (row == self.maze.shape[0]) or \
                    (col == -1) or (col == self.maze.shape[1]):
                continue
            else:
                possible_moves.append([dx, dy])

        return possible_moves

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
                if self.states[s] == 'WIN' or self.states[s] == 'DEAD':
                    transition_probabilities[s, s, a] = 1

                elif self.is_dead(s):
                    next_s = self.map['DEAD']
                    transition_probabilities[next_s, s, a] = 1

                elif self.is_win(s):
                    next_s = self.map['WIN']
                    transition_probabilities[next_s, s, a] = 1

                else:
                    mx, my = self.states[s][2], self.states[s][3]
                    next_s = self.__move(s, a)

                    px_new, py_new = self.states[next_s][0], self.states[next_s][1]
                    minotaur_moves = self.possible_minotaur_moves(s)
                    n_moves = len(minotaur_moves)
                    for move in minotaur_moves:
                        d_mx, d_my = move[0], move[1]
                        next_s = self.map[(px_new, py_new, mx + d_mx, my + d_my)]
                        transition_probabilities[next_s, s, a] = 1 / n_moves

        return transition_probabilities

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        for s in range(self.n_states):
            for a in range(0, self.n_actions):
                if self.states[s] == 'WIN' or self.states[s] == 'DEAD':
                    rewards[s, a] = 0
                    continue
                next_s = self.__move(s, a)

                # Does not move
                if s == next_s:
                    if a != self.STAY:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    else:
                        rewards[s, a] = self.STEP_REWARD

                else:
                    if self.is_win(s):
                        rewards[s, a] = self.GOAL_REWARD
                    elif self.is_dead(s):
                        rewards[s, a] = self.MINOUTAUR_REWARD
                    else:
                        rewards[s, a] = self.STEP_REWARD

        return rewards

    def simulate(self, start, policy, method, T):
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

                # Update time and state for next iteration
                t += 1
                s = next_s

        return path

    def repeat_simulate(self, start, policy, method, N, T=100):
        time_to_exit = []
        for r in range(N):
            path = self.simulate(start, policy, method, T)
            s = self.map[path[-1]]
            if self.is_win(s):
                t = math.ceil(len(path) / 2)
                time_to_exit.append(t)

        return len(time_to_exit) / N, time_to_exit

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

    # The dynamic programming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming backwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows, cols = maze.shape

    # Create figure of the size of the maze
    plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
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
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

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

    # Update the color at each frame
    for i in range(len(path)):

        ax.set_title(f'\t \t \t \t  Policy simulation \t \t \t t {i} T {len(path) - 1}'.expandtabs())
        # First clear the prev illustration, if path[i] is same as path[i-1] then it is already changed!
        # Illustration of current status
        if i > 0:
            # if path[i][0:2] != path[i-1][0:2]:
            grid.get_celld()[(path[i - 1][0:2])
            ].set_facecolor(col_map[maze[path[i - 1][0:2]]])
            grid.get_celld()[(path[i - 1][0:2])].get_text().set_text('')
            # if path[i][2:4] != path[i-1][2:4]:
            grid.get_celld()[(path[i - 1][2:4])
            ].set_facecolor(col_map[maze[path[i - 1][2:4]]])
            grid.get_celld()[(path[i - 1][2:4])].get_text().set_text('')

        # Agent illustration
        grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0:2])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2:4])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2:4])].get_text().set_text('Minotaur')

        # Position is the same and it is DEAD!
        if path[i][0:2] == path[i][2:4]:
            grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i][0:2])].get_text().set_text('DEAD')
            break  # Since nothing changes
        # Position is the same and it is WIN!
        elif maze[path[i][0:2]] == 2:
            grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_GREEN)
            grid.get_celld()[(path[i][0:2])].get_text().set_text('WIN')
            break  # Since nothing changes

        display.display(fig)
        # Save figures
        try:
            os.makedirs(f'{os.getcwd()}/animation')
        except:
            fig.savefig(f"{os.getcwd()}/animation/move{i}.png")

        display.clear_output(wait=True)
        time.sleep(1)
    # Save figures
    try:
        os.makedirs(f'{os.getcwd()}/animation')
    except IOError:
        fig.savefig(f"{os.getcwd()}/animation/move_last.png")
