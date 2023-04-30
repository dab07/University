import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Define the CNN architecture
model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(512, activation="softmax"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])
print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Using MLP
from keras.callbacks import ModelCheckpoint
import time
mlp_start = time.time()

checkpointer = ModelCheckpoint(filepath='MLP.best_weights.hdf5', verbose=1,
                               save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=20,
          validation_data=(x_valid, y_valid), callbacks=[checkpointer],
          verbose=2, shuffle=True)

mlp_end = time.time()
mlp_took = mlp_end -mlp_start
print("took %s seconds"%(mlp_took))

model.load_weights('MLP.best_weights.hdf5')
mlp_score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', mlp_score[1])


# using CNN
from keras.callbacks import ModelCheckpoint
cnn_start = time.time()
checkpointer = ModelCheckpoint(filepath='CNN.best_weights.hdf5', verbose=1,
                               save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
          validation_data=(x_valid, y_valid), callbacks=[checkpointer],
          verbose=2, shuffle=True)
cnn_end = time.time()
cnn_took = cnn_end -cnn_start
print("took %s seconds"%(cnn_took))
model.load_weights('CNN.best_weights.hdf5')
cnn_score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', cnn_score[1])

print("Time it took for the MLP to train: %f minutes. Accuracy %f " % (int(mlp_took/60),mlp_score[1]))
print("Time it took for the CNN to train: %f minutes. Accuracy %f " % (int(cnn_took/60),cnn_score[1]))