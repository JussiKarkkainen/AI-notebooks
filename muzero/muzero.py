from typing import NamedTuple, Optional
import collections
import importlib
import weights
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
import games

class MuZero:
    def __init__(self, game: str, training: bool, config: games.MuZeroConfig = None):
        self.game = game
        self.training = training
        self.config = games.make_cartpole_config()
        
    def train(self):
        print(f"Starting training on: {self.game}")
        print("__________________________________\n\n")
        shared_storage = SharedStorage()
        replay_buffer = ReplayBuffer(self.config)
        
        # TODO In the pseudocode this is done using multiple parallel jobs
        run_selfplay(self.config, shared_storage, replay_buffer)

        train_network(self.config, shared_storage, replay_buffer)

        return shared_storage.latest_network()

    def test(self):
        print(f"Starting test on: {self.game}")

    # Start training or test
    def execute(self):
        if self.training:
            latest_network = self.train()
            weights.save_weights(latest_network)

        elif not self.training:
            latest_network = weights.download_weights()
            self.test()



def run_selfplay(config: games.MuZeroConfig, shared_storage: SharedStorage, 
                 replay_buffer: ReplayBuffer):
    pass

def train_network(config: games.MuZeroConfig, shared_storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    pass
