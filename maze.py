import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

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
    GOAL_REWARD = 10
    IMPOSSIBLE_REWARD = -100
    MINOUTAUR_REWARD = -10
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
                for a in range(self.n_actions):
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

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]] < 0:
                        row, col, = self.states[next_s]
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s, a]
                        # With probability 0.5 the reward is
                        r2 = rewards[s, a]
                        # The average reward
                        rewards[s, a] = 0.5*r1 + 0.5*r2
        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    i, j, m, n = self.states[next_s]
                    # Simply put the reward as the weights o the next state.
                    rewards[s, a] = weights[i][j]
        return rewards
    
    def draw_maze(self, maze, minotaur, policy):
        # Map a color to each cell in the maze
        col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE}
    
        # Give a color to each cell
        rows, cols    = maze.shape
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]
        
        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols,rows))
    
        # Remove the axis ticks and add title title
        ax = plt.gca()
        ax.set_title('The Maze')
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        rows, cols    = maze.shape;
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)];
    
        # Create figure of the size of the maze
        plt.figure(1, figsize=(cols, rows))

        # Create a table to color
        grid = plt.table(cellText=None,
                                cellColours=colored_maze,
                                cellLoc='center',
                                loc=(0,0),
                                edges='closed');
        # Modify the height and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0/rows);
            cell.set_width(1.0/cols);
    
        col_map = {0: '*', 1: '⇠', 2: '⇢', 3: '⇡' , 4: '⇣'};
        k = minotaur[0]
        l = minotaur[1]
        for i in range(rows):
            for j in range(cols):
                if maze[i,j] == 0:
                #i = self.states[s][0]
                #j = self.states[s][1]
                    grid.get_celld()[(i,j)].get_text().set_text(col_map[policy[(self.map[(i,j,k,l)]),0]])
                #grid.get_celld()[(player_now)].get_text().set_text('Player')
        
    def simulate(self, start, policy, method, T):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state after action from policy
                path.append(self.states[next_s])
                # Add the position in the maze corresponding to the next state after random move of minotaur
                next_s = self.__move_minotaur(next_s);
                path.append(self.states[next_s])
                ## Check if terminal state
                i = self.states[next_s][0]
                j = self.states[next_s][1]
                k = self.states[next_s][2]
                l = self.states[next_s][3]
                # Have we arrived at the exit(terminal state) ?
                arrived_exit = (self.maze[i,j] == 2);
                if arrived_exit:
                    break
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter' or method =='PolIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Add the position in the maze corresponding to the next state after random move of minotaur
            next_s = self.__move_minotaur(next_s);
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while t < T:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Add the position in the maze corresponding to the next state after random move of minotaur
                next_s = self.__move_minotaur(next_s);
                path.append(self.states[next_s])
                ## Check if terminal state
                i = self.states[next_s][0]
                j = self.states[next_s][1]
                k = self.states[next_s][2]
                l = self.states[next_s][3]
                # Have we arrived at the exit(terminal state) ?
                arrived_exit = (self.maze[i,j] == 2);
                # Have we bin eaten by the minotaur(terminal state)
                eaten = (i == k) and (j == l);
                if arrived_exit or eaten:
                    break
                # Update time and state for next iteration
                t +=1;
        return path

    def sample(self, start, policy, method, T, nr):
        # returns
        # 0 if eaten
        # 1 if escaped to the exit
        # 2 if only survived the T time steps
        samples = np.zeros(nr)
        timeofexit = np.zeros(nr)
        for j in range(nr):
            path = self.simulate(start, policy, method, T);
            if path[-1][0:2] == path[-1][2:4]:
                samples[j] = 0
            elif self.maze[path[-1][0:2]] == 2:
                samples[j] = 1
            else:
                samples[j] = 2
            timeofexit[j] = len(path)
            
            
        eaten = sum(samples==0) / nr
        exited = sum(samples==1) / nr
        survived = sum(samples==2) / nr
        expected_time = sum(timeofexit) / (2*nr)
        return (eaten, exited, survived, expected_time)

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


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
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

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
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

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
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    policy = np.zeros(n_states, dtype = int)
    Bpolicy = np.ones(n_states, dtype = int)
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    
    # Iteration counter
    n   = 0;
 
    # Iterate until convergence
    while (not (policy == Bpolicy).all()):
        # Increment by one the numbers of iteration
        n += 1;
        policy = Bpolicy
        # Update the value function (policy evaluation)
        for s in range(n_states):
            V[s] = r[s, policy[s]] + gamma*np.dot(p[:,s,policy[s]],V);
        
        # Policy improvement
        for s in range(n_states):
            for a in range(n_actions):
                Q[s,a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        Bpolicy = np.argmax(Q,1);
        # Show error
        #print(n,": ",np.linalg.norm(policy - Bpolicy))
    
    # Return the obtained policy
    return V, policy;


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


                
def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: RED, 5: LIGHT_ORANGE, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    flagg = False
    for i in range(len(path)):
        player_now = path[i][0:2]
        player_before = path[i-2][0:2]
        
        minotaur_now = path[i][2:4]
        minotaur_before = path[i-2][2:4]
        
        if i == 0:
            grid.get_celld()[(minotaur_now)].set_facecolor(RED)
            grid.get_celld()[(minotaur_now)].get_text().set_text('Minotaur')
            continue
            
        # Player's turn
        if not i%2:
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
