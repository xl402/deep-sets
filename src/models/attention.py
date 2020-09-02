import tensorflow as tf


def scaled_dot_product_attention(queries, keys, values, mask):
    # attention(q, k, v) = softmax(q @ k^T / sqrt(dim(k))) @ v

    numerator = tf.matmul(queries, keys, transpose_b=True)

    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = numerator / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9)

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)
    return attention
