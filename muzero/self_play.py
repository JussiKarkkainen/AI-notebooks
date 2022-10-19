from __future__ import annotations
import math
from games import MuZeroConfig
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
        # first do single training step
        # while shared_storage.training_step < self.config.training_steps:
            # TODO Need to add seperate cases for training and inference
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
            root = Node(0)
            network_output = self.model.initial_inference(observation, init)
            self.expand_node(root, self.game.legal_actions(), 
                             self.game.to_play(), network_output)
            
            # Monte Carlo Tree Search
            MCTS(self.config, root, game_history.action_history, self.model).run()

            action = self.select_action(len(game_history.action_history), root)
            
            observation, reward, terminated, truncated, info, done = self.game.env.step(action)
            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            
            terminated = terminated
            init = False

        return game_history

    def expand_node(self, node: Node, legal_actions: list, to_play, network_output):
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        policy = {a: math.exp(network_output.policy_logits[a]) for a in legal_actions}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)
    
    def select_action(self, action_history, node: Node):
        pass

class MCTS:
    def __init__(self, config, root, action_history, network):
        self.config = config
        self.root = root
        self.action_history = action_history
        self.network = network

    def run(self):
        pass

    def select_child(self):
        pass

    def backpropagate(self):
        pass


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


