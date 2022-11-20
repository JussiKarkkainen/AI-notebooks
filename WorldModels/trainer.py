import os
import math
import jax
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import haiku as hk
import optax
import wandb
from typing import NamedTuple
from functools import partial
import models
import torch
from tqdm import tqdm
import utils
from game import Game

class VAETrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class VTrainer:
    def __init__(self, dataset, episodes):
        self.model = hk.without_apply_rng(hk.transform(self._forward))
        self.rng = jax.random.PRNGKey(seed=42)
        self.batch_size = 32
        self.train_inputs = dataset 
        self.update_weights = self._update_weights
        self.loss_fn = self._loss_fn
        self.optimizer = optax.adam(1e-4)
        self.VAEState = VAETrainingState(params=None, opt_state=None) 
        self.num_epochs = (episodes // self.batch_size) - 1
        self.encode_buffer = []
        '''
        wandb.init(project="WorldModel VAE")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        '''
    def _forward(self, x):
        net = models.ConvVAE()
        z, mu, sigma, decoded = net(x)
        return z, mu, sigma, decoded

    @partial(jax.jit, static_argnums=(0,))
    def _loss_fn(self, params, inputs):
        ''' 
        L2 distance between the input image and the reconstruction in addition to KL loss.
        '''
        z, mu, logsigma, decoded = self.model.apply(params, inputs)
        l2 = jnp.sum(optax.l2_loss(decoded, inputs))
        kld = -0.5 * jnp.sum(1 + 2*logsigma - jnp.power(mu, 2) - jnp.exp(2*logsigma))
        return l2 + kld, (z, mu, logsigma)
    
    @partial(jax.jit, static_argnums=(0,))
    def _update_weights(self, state, inputs):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0, has_aux=True)
        # ((value, aux), grads)
        loss_aux, grads = grad_fn(state.params, inputs)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return VAETrainingState(params=params, opt_state=opt_state), loss_aux[0], loss_aux[1]

    def make_initial_state(self, rng, x):
        init_params = self.model.init(rng, x)
        opt_state = self.optimizer.init(init_params)
        return VAETrainingState(params=init_params, opt_state=opt_state)

    def step(self, inputs):
        self.VAEState, loss, aux = self.update_weights(self.VAEState, inputs)
        return loss, aux

    def train(self):
        dummy_inputs = self.train_inputs[0]
        self.VAEState = self.make_initial_state(self.rng, dummy_inputs) 
        for data in tqdm(self.train_inputs):
            loss, aux = self.step(data)
            self.encode_buffer.append(aux)
            print(loss)
            #wandb.log({"loss": loss}) 

        return self.VAEState 

class MDNRNNTrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class MTrainer:
    def __init__(self, vae_dataset, mdn_lstm_actions, mdn_latent, mdn_latent_targets, episodes, v_model_params):
        self.episodes = episodes
        self.batch_size = 8
        self.latents = mdn_latent
        self.target_latents = mdn_latent_targets
        self.train_actions = mdn_lstm_actions
        self.rng = jax.random.PRNGKey(seed=42)
        self.m_model = hk.without_apply_rng(hk.transform(self._unroll_rnn))
        self.optimizer = optax.adam(1e-4)
        self.num_epochs = (episodes // self.batch_size) - 1
        self.MDNRNNState = MDNRNNTrainingState(params=None, opt_state=None)
        

        '''
        wandb.init(project="WorldModel MDNRNN")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": self.num_epochs,
            "batch_size": 32
        }
        '''
    def _unroll_rnn(self, z, a):
        batch_size = z.shape[0]
        core = models.MDN_LSTM()
        initial_state = core.initial_state(batch_size)
        inputs = (z, a)
        out, mdnlstm_state = hk.dynamic_unroll(core, inputs, initial_state, time_major=False)  
        return out, mdnlstm_state

    #@partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, z, a, y):
        # MDM-RNN predicts z_t+1 = y
        out, state = self.m_model.apply(params, z, a)
        (h, alpha, mu, logsigma) = out
        loss = 0
        for i in range(mu.shape[-1]):
            mu = mu[:, :, i].reshape(mu.shape[0], mu.shape[1], 1)
            logsigma = logsigma[:, :, i].reshape(logsigma.shape[0], logsigma.shape[1], 1)
            alpha = alpha[:, :, i].reshape(alpha.shape[0], alpha.shape[1], 1)
            loss += alpha + (-0.5 * ((y - mu) / jnp.exp(logsigma)) ** 2 - jnp.exp(logsigma) - \
                        jnp.log(jnp.sqrt(2. * jnp.pi)))
        loss = logsumexp(loss, axis=2)
        return -jnp.mean(loss)

    #@partial(jax.jit, static_argnums=(0,))
    def update_weights(self, state, inputs, targets, actions):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0)
        loss, grads = grad_fn(state.params, inputs, actions, targets)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, MDNRNNTrainingState(params=params, opt_state=opt_state)

    def make_initial_state(self, rng, z, action):
        init_params = self.m_model.init(rng, z, action)
        opt_state = self.optimizer.init(init_params)
        return MDNRNNTrainingState(params=init_params, opt_state=opt_state)
    
    def step(self, z_t, z_t1, a):
        # z_t should be of shape (batch_size, seq_len, H, W, C) 
        loss, self.MDNRNNState = self.update_weights(self.MDNRNNState, z_t, z_t1, a)
        return loss

    def train(self):
        self.MDNRNNState = self.make_initial_state(self.rng, self.latents[0], self.train_actions[0])
        # Latents should be of shape (n, bs, seq_len, h, w, c)
        for z_t, z_t1, a in tqdm(zip(self.latents, self.target_latents, self.train_actions)):
            loss = self.step(z_t, z_t1, a)
            print(loss)
            #wandb.log(["loss": loss])

        return self.MDNRNNState

class CTrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

class CTrainer:
    '''
    Original paper used CMA-ES to train the controller, this implementations uses 
    Advantage Actor Critic (A2C)
    '''
    def __init__(self, episodes, m_params, m_model, v_params, v_model, num_epochs=50):
        self.episodes = episodes
        self.m_params = m_params
        self.m_model = m_model
        self.v_params = v_params
        self.v_model = v_model
        self.c_model = hk.without_apply_rng(hk.transform(self._forward))
        self.c_state = CTrainingState(params=None, opt_state=None)        
        self.rng = jax.random.PRNGKey(seed=42)
        self.optimizer = optax.adam(5e-3)
        self.num_epochs = num_epochs
        self.entropy_beta = 1e-4

    def _forward(self, inputs):
        model = models.Controller()
        mean, var, value = model(inputs)
        sigma = jnp.sqrt(var)
        actions = mean + sigma * jax.random.normal(self.rng, mean.shape)
        a = jnp.squeeze(actions) if len(actions.shape) > 2 else actions
        steer = jnp.clip(jnp.take(a, 0, axis=1).reshape((a.shape[0], 1)), -1, 1)
        # TODO: Sigmoid?
        gas = jax.nn.sigmoid(jnp.take(a, 1, axis=1).reshape((a.shape[0], 1)))
        brake = jax.nn.sigmoid(jnp.take(a, 2, axis=1).reshape((a.shape[0], 1)))
        actions = jnp.array((steer, gas, brake)).reshape(mean.shape)
        return mean, var, actions, value 

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, z, h, targets, reward): 
        mean, var, actions, value = self.c_model.apply(params, (z, h))
        value = value.reshape(reward.shape)
        # TODO: Reward should be something else
        value_loss = jnp.mean(optax.l2_loss(value, reward))
        # TODO: Should be rt+1 + y * V(st+1) - V(st)
        adv = (reward - value).reshape(reward.shape[0], 1, 1)
        log_prob = adv * utils.logprob(mean, var, actions)
        policy_loss = jnp.mean(-log_prob)
        entropy = -(jnp.log(2*math.pi*var) + 1)/2
        entropy_loss = self.entropy_beta * jnp.mean(entropy)
        loss = value_loss + policy_loss + entropy_loss
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def update_weights(self, state, z, h, actions, rewards):
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=0)
        loss, grads = grad_fn(state.params, z, h, actions, rewards)
        updates, opt_state = self.optimizer.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)
        return loss, CTrainingState(params=params, opt_state=opt_state)

    def make_initial_state(self, z, h):
        params = self.c_model.init(self.rng, (z, h))
        opt_state = self.optimizer.init(params) 
        return CTrainingState(params=params, opt_state=opt_state)

    def train(self):
        self.c_state = self.make_initial_state(jnp.zeros([1, 32]), jnp.zeros([1, 256]))
        render = os.environ.get("RENDER")
        game = Game(render_mode="human") if render else Game()
        terminated = 0
        obs, info = game.reset()
        obs = jnp.expand_dims(utils.preprocess(obs), axis=0)
        h = jnp.zeros([1, 256])
        cumulative_reward = 0
        # TODO: Is there a better way to do this?
        for i in range(self.num_epochs):
            acts, rs, zs, hs = [], [], [], []
            epoch_reward = 0
            for j in range(50):
                z, mu, sigma, decoded = self.v_model.apply(self.v_params, obs)
                zs.append(z)
                h = jnp.reshape(h, (1, 256))
                hs.append(h)
                _, _, a, _ = self.c_model.apply(self.c_state.params, (z, h))
                a = jnp.squeeze(a)
                acts.append(a)
                a_f = [float(a[i]) for i in range(len(a))]
                obs, reward, terminated, truncated, info = game.step(a_f)
                rs.append(reward)
                obs = jnp.expand_dims(utils.preprocess(obs), axis=0)
                cumulative_reward += reward
                epoch_reward += reward
                a = jnp.reshape(a, (1, 1, 3))
                z = jnp.reshape(z, (1, 1, 32))
                (h, alpha, mu, logsigma), state = self.m_model.apply(self.m_params, z, a)
            acts, rs, zs, hs = jnp.array(acts), jnp.array(rs), jnp.array(zs), jnp.array(hs)
            loss, self.c_state = self.update_weights(self.c_state, zs, hs, acts, rs)
            print(f"Loss on epoch: {i} was: {loss}, total reward was: {epoch_reward}")
        game.close()
        return self.c_state 
