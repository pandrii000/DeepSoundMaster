import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models import convolutional_model


if __name__ == '__main__':
    # data_part = sys.argv[1]
    data_part = '0-20'
    epochs = 15
    batch_size = 1024

    X, Y = np.load(f'./data/{data_part}/X.npy'), np.load(f'./data/{data_part}/Y.npy')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    del X, Y
    gc.collect()

    # reshape to batches
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[-1], 1)
    x_test  = x_test.reshape(x_test.shape[0], x_test.shape[-1], 1)

    model = convolutional_model()
    model_name = 'convolutional_model'
    model.summary()

    model.load_weights(model_name + '/weights.h5')

    checkpoint = ModelCheckpoint(
            model_name + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5",
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max')

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, early_stopping]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    history = model.fit(x_train, y_train,
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks_list)

    model.save_weights(model_name + '/weights.h5')

    # accuracy plot
    plt.figure(figsize=(16, 4))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name + f'/epochs{epochs}_ondata_{data_part}_acc.png')

    # loss plot
    plt.figure(figsize=(16, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name + f'/epochs{epochs}_ondata_{data_part}_val.png')

    print("Done", data_part)
