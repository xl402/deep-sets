from unittest.mock import patch

import numpy as np
import tensorflow as tf
import pytest

from models import attention as attn


NUM = np.sqrt(np.log(3))
MULTIHEAD_INPUT_SHAPES = [(12, 5), (3, 4), (512, 7)]
QUERIES = tf.constant([[[0., 0., 0.],
                        [0., NUM, 0.]]])


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
    with pytest.raises(AssertionError) as e_msg:
        _ = attn.MultiHeadAttention(*shape)
    assert 'invalid number of heads' in str(e_msg)


@patch('models.attention.Dense')
def test_multihead_attention_without_masked_input(Dense):
    Dense.return_value = tf.keras.layers.Dense(3, kernel_initializer='ones')
    mha = attn.MultiHeadAttention(3, 3)
    expected = np.array([[[1.5 * NUM, 1.5 * NUM, 1.5 * NUM],
                          [2.25 * NUM, 2.25 * NUM, 2.25 * NUM]]])
    output = mha(QUERIES, QUERIES, QUERIES)
    output = output.numpy().reshape(expected.shape)
    assert np.allclose(output, expected)


@patch('models.attention.Dense')
def test_multihead_attention_with_masked_input(Dense):
    Dense.return_value = tf.keras.layers.Dense(3, kernel_initializer='ones')
    mha = attn.MultiHeadAttention(3, 3)
    expected = np.ones((1, 2, 3)) * 3 * NUM
    mask = np.array([[1, 0]])
    output = mha(QUERIES, QUERIES, QUERIES, mask=mask)
    output = output.numpy().reshape(expected.shape)
    assert np.allclose(output, expected)
