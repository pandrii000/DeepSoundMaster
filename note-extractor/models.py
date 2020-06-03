from keras.models import Sequential
from keras.layers import *


def multilayer_perceptron_model():
    model = Sequential()

    model.add(Dense(128, input_shape=(512, ), kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())

    model.add(Dense(128, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())

    model.add(Dense(128, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    
    model.add(Dense(128, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(Dense(128, kernel_initializer='glorot_normal', activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def convolutional_model():
    # https://www.researchgate.net/figure/Block-schemes-for-VGG-like-and-ResNet-like-1D-convolutional-neural-networks-Red_fig3_328136578
    # reshaped for the current feature number
    # VGG like 1D CNN

    model = Sequential()
    
    model.add(InputLayer((512, 1)))
    
    model.add(Conv1D(filters=8, kernel_size=128, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=4))
    # 1024 maps
    
    model.add(Conv1D(filters=16, kernel_size=32, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=16, kernel_size=32, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=4))
    # 512 maps
    
    model.add(Conv1D(filters=32, kernel_size=16, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=32, kernel_size=16, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    # 512 maps
    
    model.add(Conv1D(filters=64, kernel_size=8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(filters=64, kernel_size=8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool1D(pool_size=2))
    # 512 maps
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
