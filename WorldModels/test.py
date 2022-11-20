from PIL import Image
import os
import models
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from game import Game
from utils import preprocess

class Test:
    def __init__(self, v_params, v_model, m_params, m_model, c_params, c_model, dataset=None):
        self.dataset = dataset
        self.rng = jax.random.PRNGKey(seed=42)
        self.v_net = v_model 
        self.v_params = v_params
        self.m_net = m_model 
        self.m_params = m_params
        self.c_net = c_model 
        self.c_params = c_params

    def rollout(self):
        self.game = Game(render_mode="human")
        terminated = 0
        obs, info = self.game.reset()
        obs = jnp.expand_dims(preprocess(obs), axis=0)
        h = jnp.zeros([1, 256]) 
        cumulative_reward = 0
        while not terminated:
            z, mu, sigma, decoded = self.v_net.apply(self.v_params, obs)
            h = jnp.reshape(h, (1, 256))
            _, _, a, _ = self.c_net.apply(self.c_params, (z, h))
            a = jnp.squeeze(a)
            a_f = [float(a[i]) for i in range(len(a))]
            obs, reward, terminated, truncated, info = self.game.step(a_f)
            obs = jnp.expand_dims(preprocess(obs), axis=0)
            cumulative_reward += reward
            a = jnp.reshape(a, (1, 1, 3))
            z = jnp.reshape(z, (1, 1, 32))
            (h, alpha, mu, logsigma), state = self.m_net.apply(self.m_params, z, a)
        self.game.close()
        return cumulative_reward

    def test_vae(self):
        x = np.expand_dims(self.dataset.get_image(), axis=0)
        z, decoded = self.v_net.apply(self.v_params.params, x)
        out = (jnp.squeeze(decoded) * 255).astype(np.uint8)
        og = (jnp.squeeze(x) * 255).astype(np.uint8)
        plt.figure()
        plt.imshow(out)
        #plt.imshow(np.squeeze(decoded))
        plt.show()
        os.mkdir("vae_test")
        im = Image.fromarray(np.array(out))
        im.save("vae_test/vae_pred.png")
        im2 = Image.fromarray(np.array(og))
        im2.save("vae_test/vae_input.png")
