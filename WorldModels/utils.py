import skimage
import jax.numpy as jnp
import math

def preprocess(image):
    image = jnp.array(image)
    image /= 255
    image = jnp.array(skimage.transform.resize(image, (64, 64)))
    return image

def logprob(mean, var, actions):
    p1 = -((mean - actions) ** 2) / (2*jnp.clip(var, a_min=1e-3))
    p2 = -jnp.log(jnp.sqrt(2*math.pi*var))
    return p1 + p2
