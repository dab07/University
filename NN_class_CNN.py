import tensorflow as tf
from keras.utils import to_categorical
from tensorflow import keras
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist
(train_i, train_l), (test_i, test_l) = fmnist.load_data()

train_i = train_i.reshape(train_i.shape[0], 28, 28, 1)
test_i = train_i.reshape(test_i.shape[0], 28, 28, 1)
test_i = to_categorical(test_i)
test_l = to_categorical(test_l)


# normalization
def normalize(train_i, test_i):
    train_i = train_i.astype('float32')
    test_i = test_i.astype('float32')
    train_i /= 255.0
    test_i /= 255.0
normalize(train_i, test_i)


def visualize_samples(trainX):
    for i in range(9):
        plt.subplots(3, 3)
        plt.imshow(trainX[i], cmap='gray')
    plt.show()
visualize_samples(train_i)


def model_optimize(model):
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def model_train():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])
    model.summary()
    model_optimize(model)
    return model
model = model_train()



# def fit_model(model, train_i, train_l, test_i, test_l):
