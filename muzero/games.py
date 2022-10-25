import gym
import collections
from typing import Optional

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

# From the pseudocode:
# https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
# Hyperparameters from:
# https://github.com/werner-duvaud/muzero-general/blob/master/games/cartpole.py
class MuZeroConfig:
    def __init__(self, 
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 td_steps: int,
                 num_actors: int,
                 lr_init: float,
                 lr_decay_steps: float,
                 name: str,
                 known_bounds: Optional[KnownBounds] = None):

        # Network hyperparameters
        self.network = "fc"
        self.encoding_size = 8
        self.fc_representation_layers = 16  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = 16  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = 16  # Define the hidden layers in the reward network
        self.fc_value_layers = 16  # Define the hidden layers in the value network
        self.fc_policy_layers = 16  # Define the hidden layers in the policy network
        self.stacked_observations = 2 # Number of observations stacked and fed to model

        # Name
        self.game_name = name
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size

        ### Self-Play
        self.action_space_size = action_space_size
        self.action_space = list(range(action_space_size))
        self.num_actors = num_actors
        self.observation_shape = (1, 1, 4) # Channel, Height, Width
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(2)        # 1000e3
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 2                 # batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps
    
    def visit_softmax_temperature_fn(self, trained_steps):
        '''
        Temperature controls how much the agent exploits vs explores.
        Lower temperature means more greedy
        '''
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


def make_cartpole_config() -> MuZeroConfig:
    return MuZeroConfig(action_space_size=2,
                        max_moves = 500,
                        discount = 0.997,
                        dirichlet_alpha = 0.25,
                        num_simulations = 50,
                        batch_size = 128,
                        td_steps = 500,
                        num_actors = 1,
                        lr_init = 0.02,
                        lr_decay_steps = 1000,
                        name="cartpole-v1",
                        known_bounds=KnownBounds(-1, 1))



class Game:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.env.action_space.seed(42)
        self.seed = 42
   
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation, info = self.env.reset()
        return observation, info

    def render(self):
        self.env.render()

    def legal_actions(self):
        return list(range(2))
    
    def to_play(self):
        return 0

