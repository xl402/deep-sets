import tensorflow as tf
import numpy as np

from models import attention as attn


def test_scaled_dot_product_attention():
    keys = tf.constant([[10., 0., 0.],
                        [0., 10., 0.],
                        [0., 0., 10.],
                        [0., 0., 10.]])
    values = tf.constant([[1., 0.], [10., 0.], [100., 5.], [1000., 6.]])
    queries = tf.constant([[0., 10., 0.]])
    output = attn.scaled_dot_product_attention(queries, keys, values, None)
    expected = np.array([[10., 0.]])
    assert np.allclose(output.numpy(), expected)
