import jax
import jax.numpy as jnp
import haiku as hk
import optax
from games import MuZeroConfig
from typing import NamedTuple, Dict, List

Action = List[int]

class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

class Network(hk.Module):
    def __init__(self):
        super().__init__()

    # Representation + prediction
    def initial_inference(self, observation):
        pass

    # Dynamics + prediction
    def recurrent_inference(self, hidden_state, action):
        pass

    def get_weigths(self):
        pass

    


class MuZeroNetwork:
    def __init__(self, config: MuZeroConfig, init):
        self.config = config
        self.init = init
        if init:
            self.model = make_uniform_network()


def make_uniform_network():
    pass


class MuZeroFullyConnectedNet(Network):
    def __init__(self):
        pass


    def representation_fn(self, observation):
        pass

    def dynamics_fn(self, state):
        pass

    def prediction_fn(self, action):
        pass

    def initial_inference(self, observation):
        pass

    def recurrent_inference(self, hidden_state, action):
        pass

class MuZeroResidualNet(Network):
    def __init__(self):
        pass


    def representation_fn(self, observation):
        pass

    def dynamics_fn(self, state):
        pass

    def prediction_fn(self, action):
        pass

    def initial_inference(self, observation):
        pass

    def recurrent_inference(self, hidden_state, action):
        pass


