# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

def train(X, Y, args):
    dataSize = len(Y)
    trainingSize = int(dataSize * args.trainingPercentage / 100.)

    X_train, X_test = X[:trainingSize], X[trainingSize:]
    Y_train, Y_test = Y[:trainingSize], Y[trainingSize:]

    # Build Model
    model = Sequential()

    # Conv layer 1 output shape (128, 64, 64)
    model.add(Convolution2D(
        batch_input_shape=(None, 1, 64, 64),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',     # Padding method
        data_format='channels_first',
    ))
    model.add(Activation('relu'))

    # Pooling layer 1 (max pooling) output shape (128, 32, 32)
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
        data_format='channels_first',
    ))

    # Conv layer 2 output shape (256, 32, 32)
    model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
    model.add(Activation('relu'))

    # Pooling layer 2 (max pooling) output shape (256, 16, 16)
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

    # Fully connected layer 1 input shape (256 * 16 * 16) = (3136), output shape (1024)
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))

    # Fully connected layer 2 to shape (10) for 10 classes
    model.add(Dense(2))
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam,
              loss='mse',
              metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=100, batch_size=16,)

    print('\nTesting ------------')
    # Evaluate the model with the metrics we defined earlier
    loss, accuracy = model.evaluate(X_test, Y_test)

    print('\ntest loss: ', loss)
    print('test accuracy: ', accuracy)
    print(X_test[0:1].shape)
    print('test one Image: ', model.predict(X_test[0:1]))
    return model