import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
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
        # Maybe do this
        elif config.network == "residual":
            return MuZeroResidualNet(config)
        else:
            raise TypeError(f"Invalid network type: {config.network}")

class MLP(hk.Module):
    def __init__(self, num_layers, layer_size, num_outputs):
        super().__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.num_outputs = num_outputs

    def __call__(self, x):
        layers = tuple([hk.Linear(self.layer_size) for i in range(self.num_layers-1)])
        mlp = hk.Sequential([*layers, hk.Linear(self.num_outputs)])
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
        self.full_support_size = 2 * self.support_size + 1
        
        self.seed = jax.random.PRNGKey(seed=0)

        def representation_mlp_fn(x):
            mlp = MLP(3, self.fc_representation_layers, self.encoding_size)
            return mlp(x)
        self.representation_network = hk.without_apply_rng(hk.transform(representation_mlp_fn))
         
        # Value and policy are part of prediction
        def policy_mlp_fn(x):
            mlp = MLP(3, self.fc_policy_layers, self.action_space_size)
            return mlp(x)
        self.policy_network = hk.without_apply_rng(hk.transform(policy_mlp_fn))

        def value_mlp_fn(x):
            mlp = MLP(3, self.fc_value_layers, self.full_support_size) 
            return mlp(x)
        self.value_network = hk.without_apply_rng(hk.transform(value_mlp_fn))
        
        # Reward and state are part of dynamics
        def reward_mlp_fn(x):
            mlp = MLP(3, self.fc_reward_layers, self.encoding_size) 
            return mlp(x)
        self.reward_network = hk.without_apply_rng(hk.transform(reward_mlp_fn))

        def state_mlp_fn(x):
            mlp = MLP(3, self.fc_dynamics_layers, self.encoding_size) 
            return mlp(x)
        self.hidden_state_network = hk.without_apply_rng(hk.transform(state_mlp_fn))

    # Returns hidden state from observation
    def representation_fn(self, observation, init=False):
        if init:
            self.representation_params = self.representation_network.init(self.seed, observation)
        representation = self.representation_network.apply(self.representation_params, observation)
        return representation

    # Returns reward and new state from action and state
    def dynamics_fn(self, state, action, init=False):
        if init:
            r_init = np.random.randn(*state.shape).astype(np.float32)
            # TODO get rid of magic values
            s_init = np.random.randn(10, 1).astype(np.float32)
            self.reward_params = self.reward_network.init(self.seed, r_init) 
            self.state_params = self.hidden_state_network.init(self.seed, s_init)
        # One hot encode action and concatanate with state
        state, action = state.reshape(-1, 1), hk.one_hot(action, 2).reshape(-1, 1)
        x = jnp.concatenate((state, action), axis=0)
        hidden_state = self.hidden_state_network.apply(self.state_params, x)
        reward = self.reward_network.apply(self.reward_params, hidden_state)
        return reward, hidden_state
    
    # Returns policy and value for given game state
    def prediction_fn(self, hidden_state, init=False):
        if init:
            self.policy_params = self.policy_network.init(self.seed, hidden_state)
            self.value_params = self.value_network.init(self.seed, hidden_state)
        policy_logits = self.policy_network.apply(self.policy_params, hidden_state)
        value = self.value_network.apply(self.value_params, hidden_state)
        return policy_logits, value

    # representation + prediction
    def initial_inference(self, observation, init=False):
        hidden_state = self.representation_fn(observation, init)
        policy_logits, value = self.prediction_fn(hidden_state, init)
        return NetworkOutput(value=value, reward=None, 
                             policy_logits=policy_logits, hidden_state=hidden_state)
    
    # dynamics + prediction
    def recurrent_inference(self, hidden_state, action, init=False):
        reward, hidden_state = self.dynamics_fn(hidden_state, action, init)
        policy_logits, value = self.prediction_fn(hidden_state, init)
        return NetworkOutput(value=value, reward=reward, 
                             policy_logits=policy_logits, hidden_state=hidden_state)

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


