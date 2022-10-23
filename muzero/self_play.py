from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from games import MuZeroConfig, KnownBounds
import games
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from network import MuZeroNetwork

class SelfPlay:
    '''
    Gather data to replay buffer and play game during test time
    '''

    def __init__(self, game, config: MuZeroConfig, init=False):
        self.config = config
        self.init = init
        self.model = MuZeroNetwork(config, init)
        self.game = games.Game() 
        
    # Calls play_game() and stores game data to replay buffer and networks to shared storage
    def run_selfplay(self, shared_storage: SharedStorage, replay_buffer: ReplayBuffer):
        print("Starting Self-Play to fill Replay Buffer\n")
        # first do single training step, normally this would loop until replay buffer is full
        # while shared_storage.training_step < self.config.training_steps:
            # TODO: Need to add seperate cases for training and inference
        game_history = self.play_game(self.config, self.model)
        replay_buffer.save_game(game_history)
        shared_storage.increment_trainingstep()            

    # Execute Monte Carlo Tree Search to generate moves 
    def play_game(self, config: MuZeroConfig, network: MuZeroNetwork, render=False):
        game_history = GameHistory()
        observation, _ = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
                
        terminated = False
        init = True
        
        if render:
            self.game.render()

        while not terminated and len(game_history.action_history) < config.max_moves:
            # TODO: Add stacked observations
            #stacked_observation = game_history.get_stacked_observations(
            #        -1, self.config.stacked_observations, len(self.config.action_space_size))
            
            root = Node(0)
            network_output = self.model.initial_inference(observation, init)
            expand_node(root, self.game.legal_actions(), 
                             self.game.to_play(), network_output)
            
            # Monte Carlo Tree Search
            MCTS(self.game, self.config, root, game_history.action_history, self.model, init).run()

            action = self.select_action(len(game_history.action_history), root)
            
            observation, reward, terminated, truncated, info, done = self.game.env.step(action)
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            
            terminated = terminated
            init = False

        return game_history

    
    def select_action(self, action_history, node: Node):
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
        _, action = softmax_sample(visit_counts, t)
        return action
    
def expand_node(node: Node, legal_actions: list, to_play, network_output):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy_logits = jnp.squeeze(network_output.policy_logits)
    policy = {a: math.exp(policy_logits[a]) for a in legal_actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)

class MCTS:
    def __init__(self, game, config, root, action_history, network, init):
        self.game = game
        self.config = config
        self.root = root
        self.action_history = action_history
        self.network = network
        self.init = init

    def run(self):
        min_max_stats = MinMaxStats(self.config.known_bounds)
        for _ in range(self.config.num_simulations):
            node = self.root
            search_path = [node]
            
            # traverse tree until leaf node
            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                self.action_history.append(action)

            # When encountering leaf, use dynamics function to get next hidden state
            parent = search_path[-1]
            network_output = self.network.recurrent_inference(parent.hidden_state, 
                    self.action_history[-1], init=self.init)
            expand_node(node, self.game.legal_actions(), self.game.to_play(), network_output)
            self.backpropagate(search_path, support_to_scalar(network_output.value, self.config.support_size), 
                    min_max_stats)

    def select_child(self, node, min_max_stats):
        _, action, child = max((self.ucb_score(node, child, min_max_stats), \
                action, child) for action, child in node.children.items())
        return action, child
    
    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                      self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = child.reward + self.config.discount * min_max_stats.normalize(
                child.value())
        else:
            value_score = 0
        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == self.game.to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())
            value = node.reward + self.config.discount * value

def support_to_scalar(logits, support_size):
    probs = jax.nn.softmax(logits, axis=1)
    support = jnp.resize(jnp.array([x for x in range(-support_size, support_size + 1)]), probs.shape)
    x = jnp.sum(support * probs, axis=1, keepdims=True)
    x = jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * 0.001 * (jnp.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

class MinMaxStats:
    def __init__(self, known_bounds):
        self.max = known_bounds.max if known_bounds.max else -float(inf)
        self.min = known_bounds.min if known_bounds.min else float(inf)
    
    def update(self, value):
        self.maximum = max(self.max, value)
        self.minimum = min(self.min, value)

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class GameHistory:
    def __init__(self):
        self.action_history = []
        self.observation_history = []
        self.reward_history = []

    def get_stacked_observations(self, index, num_stacked, action_space_size):
        pass
