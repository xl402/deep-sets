import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense

from models.attention import MultiHeadAttention

class MLP(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.linear_1 = Dense(input_dim, activation='relu')
        self.linear_2 = Dense(input_dim, activation='relu')
        self.linear_3 = Dense(input_dim, activation='relu')

    def call(self, x):
        out = self.linear_3(self.linear_2(self.linear_1(x)))
        return out


class MultiHeadAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, num_heads: int, mlp: MLP):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multihead = MultiHeadAttention(input_dim, num_heads)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layer_norm2 = LayerNormalization(epsilon=1e-6, dtype='float32')
        self.mlp = mlp

    def call(self, source, target, mask=None):
        attention = self.multihead(source, target, target, mask)
        skip_connection = self.layer_norm1(source + attention)
        output = self.layer_norm2(skip_connection + self.mlp(skip_connection))
        return output


class SetAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim: int, num_heads: int, mlp: MLP):
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(input_dim, num_heads, mlp)

    def call(self, x, mask=None):
        return self.mab(x, x, mask)
