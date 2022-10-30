import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import skimage

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.act_buffer = []

    def save(self, observation, action):
        self.buffer.append(jnp.array(observation))
        self.act_buffer.append(jnp.array(action))
    
    def get_targets(self, batch_size):
        dataset = []
        index = 0
        for i in range((len(self.buffer) // batch_size) - 1):
            batch = []
            for j in range(batch_size):
                batch.append(self.preprocess(self.buffer[index+j+1]))
            index += batch_size
            dataset.append(batch)
        dataset = jnp.array(dataset)
        return iter(dataset)

    def get_train_inputs(self, batch_size, iterator=True):
        dataset = []
        index = 0
        for i in range((len(self.buffer) // batch_size) - 1):
            batch = []
            for j in range(batch_size):
                batch.append(self.preprocess(self.buffer[index+j]))
            index += batch_size
            dataset.append(batch)
        dataset = jnp.array(dataset)
        return iter(dataset) if iterator else dataset

    def get_train_actions(self, batch_size, iterator=True):
        dataset = []
        index = 0
        for i in range((len(self.act_buffer) // batch_size) - 1):
            batch = []
            for j in range(batch_size):
                batch.append(self.act_buffer[index+j])
            index += batch_size
            dataset.append(batch)
        dataset = jnp.array(dataset)
        return iter(dataset) if iterator else dataset

    def preprocess(self, image):
        image = jnp.array(skimage.transform.resize(image, (64, 64)))
        return image
