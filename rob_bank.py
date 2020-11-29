import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math

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
    CAUGHT_REWARD = -50
    BANK_REWARD = 10
    IMPOSSIBLE_REWARD = -1e-10
    STEP_REWARD = 0

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
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
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

        # Reinitialised when caught by the police
        if i == m and j == n:
            return self.map[(0, 0, 1, 2)]

        # Is the future position an impossible one ?
        row = i + self.actions[action][0]
        col = j + self.actions[action][1]

        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1])

        # Based on the impossibility check return the next state.
        if hitting_maze_walls:
            return state
        else:
            return self.map[(row, col, m, n)]

    def __move_police(self, state):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """

        # Compute the future position given current (state, action)
        i = self.states[state][0]
        j = self.states[state][1]
        m = self.states[state][2]
        n = self.states[state][3]

        # On the same row
        if i == m and j < n:
            action = np.random.choice([self.MOVE_UP, self.MOVE_DOWN, self.MOVE_LEFT])[0]
        elif i == m and j > n:
            action = np.random.choice([self.MOVE_UP, self.MOVE_DOWN, self.MOVE_RIGHT])[0]
        elif j == n and i < m:
            action = np.random.choice([self.MOVE_LEFT, self.MOVE_UP, self.MOVE_RIGHT])[0]
        elif j == n and i > m:
            action = np.random.choice([self.MOVE_LEFT, self.MOVE_DOWN, self.MOVE_RIGHT])[0]
        # right bottom
        elif i < m and j < n:
            action = np.random.choice([self.MOVE_UP, self.MOVE_LEFT])[0]
        # left bottom
        elif i < m and j > n:
            action = np.random.choice([self.MOVE_UP, self.MOVE_RIGHT])[0]
        # right upper
        elif i > m and j < n:
            action = np.random.choice([self.MOVE_DOWN, self.MOVE_LEFT])[0]
        # left upper
        else:
            action = np.random.choice([self.MOVE_DOWN, self.MOVE_RIGHT])[0]

        # Is the future position an impossible one ?
        row = m + self.actions[action][0]
        col = n + self.actions[action][1]
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1])

        # Based on the impossibility check return the next state.
        if hitting_maze_walls:
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

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(0, self.n_actions):
                next_s = self.__move(s, a)

                i = self.states[next_s][0]
                j = self.states[next_s][1]
                m = self.states[next_s][2]
                n = self.states[next_s][3]

                in_bank = (i == 0 and j == 0) or (i == 2 and j == 0) or (i == 0 and j == 5) or (i == 2 and j == 5)

                caught = i == m and j == n

                if s == next_s and a != self.STAY:
                    rewards[s, a] = self.IMPOSSIBLE_REWARD

                # Reward for being caught by the police
                elif caught:
                    rewards[s, a] = self.CAUGHT_REWARD

                # Reward for robbing bank
                elif in_bank:
                    rewards[s, a] = self.BANK_REWARD

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


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def policy_iteration(env, gamma):
    """ Solves the shortest path problem using policy iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    policy = np.zeros(n_states, dtype=int)
    Bpolicy = np.ones(n_states, dtype=int)
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))

    # Iteration counter
    n = 0;

    # Iterate until convergence
    while (not (policy == Bpolicy).all()):
        # Increment by one the numbers of iteration
        n += 1;
        policy = Bpolicy
        # Update the value function (policy evaluation)
        for s in range(n_states):
            V[s] = r[s, policy[s]] + gamma * np.dot(p[:, s, policy[s]], V);

        # Policy improvement
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V);
        Bpolicy = np.argmax(Q, 1);
        # Show error
        # print(n,": ",np.linalg.norm(policy - Bpolicy))

    # Return the obtained policy
    return V, policy;


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: LIGHT_GREEN, 2: LIGHT_RED, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

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
    rows, cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows);
        cell.set_width(1.0 / cols);

    # Update the color at each frame
    flagg = False
    for i in range(len(path)):
        player_now = path[i][0:2]
        player_before = path[i - 2][0:2]

        minotaur_now = path[i][2:4]
        minotaur_before = path[i - 2][2:4]

        if i == 0:
            grid.get_celld()[(minotaur_now)].set_facecolor(RED)
            grid.get_celld()[(minotaur_now)].get_text().set_text('Minotaur')
            continue

        # Player's turn
        if not i % 2:
            if player_now == minotaur_now:
                # The player goes to the minotaur and gets eaten
                grid.get_celld()[(player_now)].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(player_now)].get_text().set_text('Player')
                grid.get_celld()[(player_now)].set_facecolor(RED)
                grid.get_celld()[(player_now)].get_text().set_text('EATEN')
                flagg = True
            elif maze[player_now] == 2:
                # The player escapes
                grid.get_celld()[(player_now)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(player_now)].get_text().set_text('FINISH')
                flagg = True
            else:
                # The player does normal move
                grid.get_celld()[(player_now)].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(player_now)].get_text().set_text('Player')
            if not player_now == player_before:
                # Reset the color of last position
                grid.get_celld()[(player_before)].set_facecolor(col_map[maze[player_before]])
                grid.get_celld()[(player_before)].get_text().set_text('')
                # Minotaur's turn
        else:
            if player_now == minotaur_now:
                # The minotaur eats the player
                grid.get_celld()[(player_now)].set_facecolor(RED)
                grid.get_celld()[(player_now)].get_text().set_text('EATEN')
                flagg = True
            else:
                # The minotaur does a normal move
                grid.get_celld()[(minotaur_now)].set_facecolor(RED)
                grid.get_celld()[(minotaur_now)].get_text().set_text('Minotaur')
            if not minotaur_now == minotaur_before:
                # Reset the color of last position
                grid.get_celld()[(minotaur_before)].set_facecolor(col_map[maze[minotaur_before]])
                grid.get_celld()[(minotaur_before)].get_text().set_text('')

        display.display(fig)
        display.clear_output(wait=True)
        if flagg:
            break
        time.sleep(0.3)
