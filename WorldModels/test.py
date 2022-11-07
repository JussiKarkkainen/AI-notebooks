from PIL import Image
import os
import models
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from game import Game

class Test:
    def __init__(self, v_params, v_model, m_params, m_model, c_params, c_model, dataset=None):
        self.dataset = dataset
        self.rng = jax.random.PRNGKey(seed=42)
        self.v_net = hk.without_apply_rng(hk.transform(self.forward))
        self.vae_params = vae_model
        self.init_params = self.net.init(self.rng, jnp.zeros((1, 64, 64, 3)))
        self.m_net = hk.without_apply_rng(hk.transform(self.m_forward))
        self.m_params = m_params
        self.c_net = hk.without_apply_rng(hk.transform(self.c_forward))
        self.c_params = c_params
        self.game = Game(render_mode="human")

    def v_forward(self, x):
        vae = models.ConvVAE()
        z, mu, std, decoded = vae(x)
        return z, decoded
    
    def rollout(self):
        terminated = 0
        obs, info = jnp.expand_dims(self.preprocess(self.game.reset()), axis=0)
        h = self.m_net.initial_state()
        cumulative_reward = 0
        while not terminated:
            z, decoded = self.v_net.apply(self.v_params.params_, obs)
            a = self.c_net.apply(self.c_params.params, (z, h))
            obs, reward, terminated, truncated, info = self.game.step(a)
            obs = jnp.expand_dims(self.preprocess(obs), axis=0)
            cumulative_reward += reward
            h = self.m_net.apply(self.m_params.params, (a, z, h))
        self.game.close()
        return cumulative_reward

    def preprocess(self, obs):
        image = jnp.array(image)
        image /= 255
        image = jnp.array(skimage.transform.resize(image, (64, 64)))
        return image

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
