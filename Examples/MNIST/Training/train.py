# Train a variant of LeNet-5 on MNIST.
#
# This gets 99.21% test set accuracy (after approx. 10 epochs).

import os.path
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

image_height, img_width = 28, 28
num_classes = 10
learning_rate = 1e-4
model_name = "mnist.h5"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], image_height, img_width, 1)
x_test = x_test.reshape(x_test.shape[0], image_height, img_width, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if os.path.isfile(model_name):
    print("*** Resuming training ***")
    model = load_model(model_name)
else:
    model = Sequential()
    model.add(Conv2D(filters=20, kernel_size=(5, 5), 
                     padding="same", activation="relu",
                     input_shape=(image_height, img_width, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=50, kernel_size=(5, 5),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=320, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=learning_rate),
                metrics=["accuracy"])

model.summary()

model.fit(x_train, y_train,
          batch_size=15,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))

model.save(model_name)
