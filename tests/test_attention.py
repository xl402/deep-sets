from mock import patch

import numpy as np
import tensorflow as tf
import pytest

from models import attention as attn


MULTIHEAD_INPUT_SHAPES = [(12, 5), (3, 4), (512, 7)]


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
                        [0., 10., 0.]])
    values = tf.constant([[5., 9.], [2., 6.], [5., 3.], [10000., 10000.]])
    mask = np.array([[0, 0, 0, 1]])
    output = attn.scaled_dot_product_attention(queries, keys, values, mask)
    expected = np.array([[2., 6.]])
    assert np.allclose(output.numpy(), expected)


@pytest.mark.parametrize('shape', MULTIHEAD_INPUT_SHAPES)
def test_raises_for_invalid_number_of_heads_for_multihead_attention(shape):
    with pytest.raises(AssertionError) as e:
        _ = attn.MultiHeadAttention(*shape)
    assert 'invalid number of heads' in str(e)


@patch('models.attention.Dense')
def test_multihead_attention_without_masked_input(Dense):
    Dense.return_value = tf.keras.layers.Dense(3, kernel_initializer='ones')
    query = tf.constant([[0, 0., 0.],
                         [0., 1., 0.],
                         [0., 1., 1.],
                         [1., 1., 1.]])
    mha = attn.MultiHeadAttention(3, 3)
    expected = np.array([[0, 0., 0.],
                         [3., 3., 3.],
                         [6., 6., 6.],
                         [9., 9., 9.]])
    output = mha(query, query, query)
    output = output.numpy().reshape(expected.shape)
    assert np.allclose(output, expected)
