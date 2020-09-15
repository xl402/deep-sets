import tensorflow as tf
from tensorflow.keras.layers import Dense


def scaled_dot_product_attention(queries, keys, values, mask):
    """
    attention(q, k, v) = softmax(q @ k^T / sqrt(dim(k))) @ v
    """
    product = tf.matmul(queries, keys, transpose_b=True)
    keys_dim = tf.cast(keys.shape[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_product, axis=-1)
    attention = tf.matmul(attention_weights, values)
    return attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, output_dim, num_heads: int, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        total_output_dim = output_dim * num_heads
        self.query_matrix = Dense(total_output_dim, **kwargs)
        self.key_matrix = Dense(total_output_dim, **kwargs)
        self.value_matrix = Dense(total_output_dim, **kwargs)

        self.dense_out = Dense(output_dim, **kwargs)

    def split_heads(self, x, batch_size):
        output_shape = (batch_size, -1, self.num_heads, self.output_dim)
        x = tf.reshape(x, output_shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]

        queries = self.query_matrix(queries)
        keys = self.key_matrix(keys)
        values = self.value_matrix(values)

        queries = self.split_heads(queries, batch_size)
        keys = self.split_heads(keys, batch_size)
        values = self.split_heads(values, batch_size)

        attention = scaled_dot_product_attention(queries, keys, values, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        output_shape = (batch_size, -1, self.output_dim * self.num_heads)
        concat_attention = tf.reshape(attention, output_shape)
        output = self.dense_out(concat_attention)
        return output
