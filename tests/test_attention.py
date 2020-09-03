import tensorflow as tf
import numpy as np

from models import attention as attn


def test_scaled_dot_product_attention_without_masked_input():
    queries = tf.constant([[0., 10., 0.]])
    keys = tf.constant([[3., 0., 0.],
                        [0., 10., 0.],
                        [0., 0., 4.],
                        [0., 0., 1.]])
    values = tf.constant([[5., 9.], [2., 6.], [5., 3.], [5., 3.]])
    output = attn.scaled_dot_product_attention(queries, keys, values, None)
    expected = np.array([[2., 6.]])
    assert np.allclose(output.numpy(), expected)


def test_scaled_dot_product_attention_with_masked_input():
    queries = tf.constant([[0., 10., 0.]])
    keys = tf.constant([[3., 0., 0.],
                        [0., 10., 0.],
                        [0., 0., 4.],
                        [0., 0., 0.]])
    values = tf.constant([[5., 9.], [2., 6.], [5., 3.], [0., 0.]])
    mask = np.array([[0, 0, 0, 1]])
    output = attn.scaled_dot_product_attention(queries, keys, values, mask)
    expected = np.array([[2., 6.]])
    assert np.allclose(output.numpy(), expected)
