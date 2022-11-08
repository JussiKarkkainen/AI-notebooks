# Collect 10 000 random rollouts from env to train vae
import os
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
    
    def save(self, buf, name=None):
        vae_dataset, mdn_lstm_actions, mdn_latents, mdn_latent_targets = buf
        path = f"data/{name}.npz"
        if not os.path.exists("data"): 
            os.mkdir("data") 
        if not os.path.exists(path):
            open(path, 'w').close()
        if name == "vae_dataset":
            np.savez(path, vae_dataset=vae_dataset) 
        elif name == "mdnrnn":
            np.savez(path, mdn_lstm_actions=mdn_lstm_actions, 
                           mdn_latents=mdn_latents, 
                           mdn_latent_targets=mdn_latent_targets)

        return path

    def load(self, path):
        datasets = np.load(path, allow_pickle=True)
        if path == "data/vae_dataset.npz":
            vae_dataset = datasets['vae_dataset']
            return vae_dataset
        else:
            mdn_lstm_actions = datasets['mdn_lstm_actions']
            mdn_latents = datasets['mdn_latents']
            mdn_latent_targets = datasets['mdn_latent_targets']
            return mdn_lstm_actions, mdn_latents, mdn_latent_targets
