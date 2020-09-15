from unittest.mock import patch

import numpy as np
import tensorflow as tf
import pytest

from models import attention as attn


NUM = np.sqrt(np.log(3))
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


def test_multihead_attention_without_masked_input():
    mha = attn.MultiHeadAttention(3, 1, 30, kernel_initializer='ones')
    expected = np.array([[[15 * NUM],
                          [22.5 * NUM]]])
    output = mha(QUERIES, QUERIES, QUERIES)
    output = output.numpy().reshape(expected.shape)
    assert np.allclose(output, expected)


def test_multihead_attention_with_masked_input():
    mha = attn.MultiHeadAttention(3, 1, 30, kernel_initializer='ones')
    expected = np.ones((1, 2, 1)) * 30. * NUM
    mask = np.array([[1, 0]])
    output = mha(QUERIES, QUERIES, QUERIES, mask=mask)
    output = output.numpy().reshape(expected.shape)
    assert np.allclose(output, expected)
