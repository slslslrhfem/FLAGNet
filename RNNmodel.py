import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import regularizers

def RNN_training(classifier,bar_matrix_list3):
    RNNx_train = []
    RNNy_train = []
    RNNx_test = []
    RNNy_test = []
    for i in range(len(bar_matrix_list3)):
        nowseq = []
        nowmat = np.array(bar_matrix_list3[i])
        nowbars = classifier.predict(nowmat.reshape(len(nowmat), 24, 24, 1))
        for j, bars in enumerate(nowbars):
            if (j == len(nowbars) - 1):
                if (i > 4500):
                    RNNy_test.append(np.argmax(bars))
                else:
                    RNNy_train.append(np.argmax(bars))
            else:
                nowseq.append(np.argmax(bars))
        if (i > 4500):
            RNNx_test.append(nowseq)
        else:
            RNNx_train.append(nowseq)
    RNNy_test = to_categorical(np.array(RNNy_test))
    RNNy_train = to_categorical(np.array(RNNy_train))
    # fix random seed for reproducibility
    numpy.random.seed(7)
    top_words = 13
    # truncate and pad input sequences
    max_review_length = 0
    for seq in RNNx_train:
        if (len(seq) > max_review_length):
            max_review_length = len(seq)
    for seq in RNNx_test:
        if (len(seq) > max_review_length):
            max_review_length = len(seq)
    RNNx_train = sequence.pad_sequences(RNNx_train, maxlen=max_review_length)
    RNNx_test = sequence.pad_sequences(RNNx_test, maxlen=max_review_length)
    model_path = 'RNNresult/' + 'RNN.h5'

    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy',
                                    verbose=1, save_best_only=True)
    callbacks = [cb_checkpoint]
    embedding_vecor_length = 32
    timesteps = 8

    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, return_sequences=True,
                   input_shape=(timesteps, 100)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(100))  # return a single vector of dimension 32
    model.add(Dropout(0.2))
    model.add(Dense(13, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    hist = model.fit(RNNx_train, RNNy_train, validation_data=(RNNx_test, RNNy_test), epochs=30, batch_size=32,
                     callbacks=callbacks)