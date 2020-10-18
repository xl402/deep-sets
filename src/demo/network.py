import tensorflow as tf
from tensorflow.keras.layers import Dense

from models.blocks import MLP, SetAttentionBlock


class SetTransformer(tf.keras.Model):
    def __init__(self, input_dim, output_dim, num_heads):
        super(SetTransformer, self).__init__()
        self.linear = Dense(input_dim, activation='relu')
        self.sab1 = SetAttentionBlock(input_dim, input_dim, num_heads, MLP(input_dim))
        self.sab2 = SetAttentionBlock(input_dim, output_dim, num_heads, MLP(output_dim))

    def call(self, x, mask=None):
        x_1 = self.sab1(x, mask)
        x_2 = self.sab2(self.linear(x_1), mask)
        return x_2
