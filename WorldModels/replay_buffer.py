import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import skimage

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def save(self, observation):
        self.buffer.append(jnp.array(observation))
    
    def get_train_inputs(self, batch_size):
        dataset = []
        for i in range((len(self.buffer) // batch_size)):
            batch = []
            for j in range(batch_size):
                batch.append(self.preprocess(self.buffer[j]))
            dataset.append(batch)
        dataset = jnp.array(dataset)
        return iter(dataset)

    def preprocess(self, image):
        image = jnp.array(skimage.transform.resize(image, (64, 64)))
        return image
