import numpy as np

from Evaluation import evaluation
from compiled_tcn import compiled_tcn
from keras.layers import *
from keras.models import *

def Model_TCN(train_data, train_target, test_data, test_target):
    Model = compiled_tcn(num_feat=train_data.shape[1], num_classes=train_target.shape[1], nb_filters=5, kernel_size=3,
                         dilations=[1, 2, 4, 8, 16, 32, 64],nb_stacks=2,opt='adam', max_len=1)

    Models = Sequential()
    # model.add(Embedding(n_unique_words, 128, input_length=maxlen))
    Models.add(Dropout(0.5))
    Models.add(Dense(1, activation='sigmoid'))
    Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    Models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    Models.fit(train_data, train_target.astype('float'), epochs=5, batch_size=128)
    pred = Models.predict(test_data)
    Eval = evaluation(pred.astype('int'), test_target)
    return np.asarray(Eval).ravel()



