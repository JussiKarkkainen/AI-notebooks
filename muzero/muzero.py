import sys
import argparse
from typing import NamedTuple, Optional
import collections
import importlib
import weights
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from trainer import Trainer
import games

class MuZero:
    def __init__(self, game: str, training: bool, config: games.MuZeroConfig = None):
        self.game = game
        self.training = training
        if game == 'cartpole-v1':
            self.config = games.make_cartpole_config()
        else:
            self.config = config

    def train(self):
        print(f"Starting training on: {self.game}")
        print("__________________________________\n\n")
        shared_storage = SharedStorage()
        replay_buffer = ReplayBuffer(self.config)
        
        # TODO In the pseudocode this is done using multiple parallel jobs
        SelfPlay(self.game, self.config, init=True).run_selfplay(shared_storage, replay_buffer)

        Trainer(self.config).train_network(shared_storage, replay_buffer)

        return shared_storage.latest_network()

    def test(self):
        print(f"Starting test on: {self.game}")
        print("__________________________________\n\n")

    # Start training or test
    def execute(self):
        if self.training:
            latest_network = self.train()
            weights.save_weights(latest_network)

        elif not self.training:
            latest_network = weights.download_weights()
            self.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuZero")
    parser.add_argument('--train', help='choose game to train on: Supported games include "cartpole-v1"')
    parser.add_argument('--test', help='choose game to test on: Supported games include "cartpole-v1"')
    args = parser.parse_args()
    
    
    if args.train:
        muzero = MuZero(args.train, training=True)
        muzero.execute()
    elif args.test:
        muzero = MuZero(args.test, training=False)
        muzero.execute()
    elif args.train and args.test:
        print("Invalid arguments, choose either train or test but not both")
    else:
        # Default is test
        muzero = MuZero(args.test, training=False)
        muzero.execute()

