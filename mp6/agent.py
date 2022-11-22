import numpy as np
import utils


def position_value(a, b):
    if a == b:
        return 0
    else:
        return 1 if a < b else 2


def adjoin_value(a, b):
    if b == a - 1 or a == 1:
        return 1
    if b == a + 1 or a == -1:
        return 2
    return 0


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self._train = True
        self.points = 0
        self.s = None
        self.a = None

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def act(self, environment, points, dead):
        """
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        """
        s_prime = self.generate_state(environment)

        # TODO: write your function here
        a_star = self.get_a_star(s_prime)
        if self._train:
            r = 1.0 if points > self.points else (-1.0 if dead else -0.1)
            if self.s is not None and self.a is not None:
                self.N[self.s][self.a] += 1
                alpha = self.C / (self.C + self.N[self.s][self.a])
                self.Q[self.s][self.a] += alpha * (r + self.gamma * np.max(self.Q[s_prime]) - self.Q[self.s][self.a])

        if dead:
            self.reset()
        else:
            self.points = points
            self.s = s_prime
            self.a = a_star

        return a_star

    def get_a_star(self, s):
        f = np.array(self.Q[s])
        if self._train:
            f[self.N[s] < self.Ne] = 1
        return np.max(np.argwhere(f == np.max(f)))

    @staticmethod
    def generate_state(environment):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
        snake_body = set(snake_body)
        return (
            position_value(food_x, snake_head_x),
            position_value(food_y, snake_head_y),
            adjoin_value(snake_head_x, utils.DISPLAY_WIDTH - 1),
            adjoin_value(snake_head_y, utils.DISPLAY_HEIGHT - 1),
            1 if (snake_head_x, snake_head_y - 1) in snake_body else 0,
            1 if (snake_head_x, snake_head_y + 1) in snake_body else 0,
            1 if (snake_head_x - 1, snake_head_y) in snake_body else 0,
            1 if (snake_head_x + 1, snake_head_y) in snake_body else 0
        )
