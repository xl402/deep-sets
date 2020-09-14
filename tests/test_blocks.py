import itertools

import numpy as np
import tensorflow as tf
import pytest

from models import blocks


INPUT_DIMS = [10, 20]
SEQ_LENGTHS = [1, 42, 69]


def test_mlp():
    input_dim, seq_len = INPUT_DIMS[0], SEQ_LENGTHS[0]
    mlp = blocks.MLP(input_dim)
    shape = (1, seq_len, input_dim)
    y = tf.random.uniform(shape, dtype=tf.float64)
    out = mlp(y).numpy()
    assert np.allclose(out.shape, np.array(shape))


def test_multihead_attention_block():
    input_dim, seq_len = INPUT_DIMS[0], SEQ_LENGTHS[0]
    mlp = blocks.MLP(input_dim)
    mab = blocks.MultiHeadAttentionBlock(input_dim, 5, mlp)
    shape = (1, seq_len, input_dim)
    y = tf.random.uniform(shape, dtype=tf.float64)
    out = mab(y, y).numpy()
    assert np.allclose(out.shape, np.array(shape))


@pytest.mark.parametrize('input_dim, seq_len',
                         itertools.product(INPUT_DIMS, SEQ_LENGTHS))
def test_set_attention_block_is_permutation_equivariant(input_dim, seq_len):
    num_heads = 5
    mlp = blocks.MLP(input_dim)
    sab = blocks.SetAttentionBlock(input_dim, num_heads, mlp)

    y = tf.random.uniform((1, seq_len, input_dim), dtype=tf.float64)
    unshuffled_output = sab(y)

    # shuffle along sequence dimension
    indices = tf.range(start=0, limit=y.shape[1], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_y = tf.gather(y, shuffled_indices, axis=1)
    expected = tf.gather(unshuffled_output, shuffled_indices, axis=1)
    output = sab(shuffled_y)

    assert np.allclose(output.numpy(), expected.numpy(), atol=1e-6)


@pytest.mark.parametrize('input_dim, seq_len',
                         itertools.product(INPUT_DIMS, SEQ_LENGTHS))
def test_masked_set_attention_block_is_permutation_equivariant(input_dim, seq_len):
    num_heads = 5
    mlp = blocks.MLP(input_dim)
    sab = blocks.SetAttentionBlock(input_dim, num_heads, mlp)

    y = tf.random.uniform((1, seq_len, input_dim), dtype=tf.float64)
    mask = np.random.randint(0, 2, (1, seq_len))
    unshuffled_output = sab(y, mask=mask)

    # shuffle along sequence dimension
    indices = tf.range(start=0, limit=y.shape[1], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_y = tf.gather(y, shuffled_indices, axis=1)
    shuffled_mask = np.take(mask, shuffled_indices.numpy(), axis=1)
    expected = tf.gather(unshuffled_output, shuffled_indices, axis=1)
    output = sab(shuffled_y, mask=shuffled_mask)

    assert np.allclose(output.numpy(), expected.numpy(), atol=1e-6)


@pytest.mark.parametrize('input_dim, seq_len',
                         itertools.product(INPUT_DIMS, SEQ_LENGTHS))
def test_masked_set_attention_block_is_independent_of_masked_data(input_dim, seq_len):
    num_heads = 5
    mlp = blocks.MLP(input_dim)
    sab = blocks.SetAttentionBlock(input_dim, num_heads, mlp)

    y = tf.random.uniform((1, seq_len, input_dim), dtype=tf.float64)
    mask = np.random.randint(0, 2, (1, seq_len))
    original_output = sab(y, mask=mask).numpy()

    # change input data that has been masked
    mask_idx = np.where(mask)
    y = y.numpy()
    y[mask_idx] = 0
    masked_output = sab(tf.convert_to_tensor(y), mask=mask).numpy()

    valid_idx = np.where(mask == 0)
    expected = original_output[valid_idx]
    output = masked_output[valid_idx]
    assert np.allclose(output, expected, atol=1e-6)
