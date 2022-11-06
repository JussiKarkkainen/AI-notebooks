# Collect 10 000 random rollouts from env to train vae
from PIL import Image
import numpy as np
from game import Game
from replay_buffer import ReplayBuffer

class Dataset:
    def __init__(self):
        self.game = Game()
        self.episodes = 33 
        self.seq_len = 17
        self.buf = ReplayBuffer(self.seq_len)

    def get_random_action(self):
        return self.game.env.action_space.sample()

    def rollout(self):
        for i in range(self.episodes):
            obs = self.game.env.reset()
            for t in range(self.seq_len):
                action = self.get_random_action()
                observation, reward, terminated, truncated, info = self.game.env.step(action)
                self.buf.save(observation, action, reward)
        return self.buf, self.episodes*self.seq_len

