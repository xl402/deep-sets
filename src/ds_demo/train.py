from tensorflow import keras

from ds_demo.data import NearestNeighbourGenerator as NNGenerator
from ds_demo.network import SetTransformer


if __name__ == '__main__':
    dim = 2
    max_len = 10
    train_len = 1000
    val_len = 200
    batch_size = 100

    train_generator = NNGenerator(dim, max_len, train_len, batch_size, 0)
    val_generator = NNGenerator(dim, max_len, val_len, batch_size, 1)

    model = SetTransformer(dim, 1, 20)
    model.compile(optimizer='Adam', loss=keras.losses.MeanSquaredError())
    model.fit(train_generator,
              validation_data=val_generator,
              epochs=10)
