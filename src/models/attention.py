import tensorflow as tf


def scaled_dot_product_attention(queries, keys, values, mask):
    """
    attention(q, k, v) = softmax(q @ k^T / sqrt(dim(k))) @ v
    """
    product = queries @ tf.transpose(keys)
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9)

    numerator = tf.nn.softmax(scaled_product, axis=-1)
    attention = numerator @ values

    return attention
