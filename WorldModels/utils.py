import skimage
import jax.numpy as jnp

def preprocess(image):
    image = jnp.array(image)
    image /= 255
    image = jnp.array(skimage.transform.resize(image, (64, 64)))
    return image

