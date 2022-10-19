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

class MuZeroNetwork:
    def __new__(cls, config: MuZeroConfig, init):
        if config.network == "fc":
            return MuZeroFullyConnectedNet(config)
        elif config.network == "residual":
            return MuZeroResidualNet(config)
        else:
            raise TypeError(f"Invalid network type: {config.network}")


def make_uniform_network():
    
    pass

class MLP(hk.Module):
    def __init__(self, num_layers, layer_size, num_outputs):
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.num_outputs = num_outputs

    def __call__(self, x):
        mlp = hk.Sequential([[hk.Linear(self.layer_size) for i in range(self.num_layers-1)],
                              hk.Linear(self.num_outputs)])
        return mlp(x)


class MuZeroFullyConnectedNet:
    def __init__(self, config):
        self.observation_shape = config.observation_shape
        self.action_space_size = config.action_space_size
        self.encoding_size = config.encoding_size
        self.fc_reward_layers = config.fc_reward_layers
        self.fc_value_layers = config.fc_value_layers
        self.fc_policy_layers = config.fc_policy_layers
        self.fc_representation_layers = config.fc_representation_layers
        self.fc_dynamics_layers = config.fc_dynamics_layers
        self.support_size = config.support_size
        self.full_support_size = 2 * support_size + 1

        self.representation_mlp = MLP(2, self.representation_layers, self.encoding_size)
        # Value and policy are part of prediction
        self.policy_mlp = MLP(2, self.policy_layers, self.action_space_size)
        self.value_mlp = MLP(2, self.value_layers, self.full_support_size) 
        self.dynamics_mlp = MLP(2, self.dynamics_layers, self.encoding_size) 

        self.seed = jax.random.PRNGKey(seed=0)

    # Returns hidden state from observation
    def representation_fn(self, observation):
        def mlp_fn(x):
            mlp = self.representation_mlp 
            return mlp(x)
        
        mlp = hk.without_apply_rng(hk.transform(mlp_fn))
        params = mlp.init(seed, observation)
        representation = mlp.apply(params, observation)
        return representation

    # Returns reward and new state from action and state
    def dynamics_fn(self, state, action):
        pass
    
    # Returns policy and value for given game state
    def prediction_fn(self, hidden_state):
        def policy_mlp_fn(x):
            mlp = self.policy_mlp 
            return mlp(x)
        def value_mlp_fn(x):
            mlp = self.value_mlp 
            return mlp(x)
        
        policy_mlp = hk.without_apply_rng(hk.transform(policy_mlp_fn))
        value_mlp = hk.without_apply_rng(hk.transform(value_mlp_fn))
        
        policy_params = policy_mlp.init(seed, hidden_state)
        value_params = value_mlp.init(seed, hidden_state)

        policy = policy_mlp.apply(params, hidden_state)
        value = value_mlp.apply(params, hidden_state)

        policy_logits = policy_mlp_fn(hidden_state)
        value = value_mlp_fn(hidden_state)
        return NetworkOutput(policy, value)

    # representation + prediction
    def initial_inference(self, observation):
        representation = self.representation_fn(observation)
        prediction = self.prediction(representation)
        return NetworkOutput(
    
    # dynamics + prediction
    def recurrent_inference(self, hidden_state, action):
        pass

class MuZeroResidualNet:
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


