# Collect 10 000 random rollouts from env to train vae
from PIL import Image
import numpy as np
from game import Game
from replay_buffer import ReplayBuffer

class Dataset:
    def __init__(self):
        self.game = Game()
        self.episodes = 128  # for now, batch_size = 32 -> need to have 10080 examples = 1080 episodes
        self.steps = 10 # for now
        self.buf = ReplayBuffer()

    def get_random_action(self):
        return self.game.env.action_space.sample()

    def rollout(self):
        for i in range(self.episodes):
            obs = self.game.env.reset()
            for t in range(self.steps):
                action = self.get_random_action()
                observation, reward, terminated, truncated, info = self.game.env.step(action)
                self.buf.save(observation, action)
        return self.buf, self.episodes*self.steps

