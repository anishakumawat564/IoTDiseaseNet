from keras.layers import *
from keras.models import *
from keras import backend as K
import numpy as np
from Evaluation import evaluation


class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention,self).__init__()

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1))
        super(attention,self).build(input_shape)


def call(self, x):
    e = K.tanh(K.dot(x,self.W)+self.b)
    a = K.softmax(e, axis=1)
    output = x*a
    if self.return_sequences:

        return output
    return K.sum(output, axis=1)

def build(self, input_shape):
    self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                             initializer="normal")
    self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                             initializer="zeros")

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x

def Model_Attention_LSTM(Train_Data, Train_Target, Test_Data, Test_Target):
    trainX = np.resize(Train_Data, (Train_Data.shape[0], 1, Train_Data.shape[1]))
    testX = np.resize(Test_Data, (Test_Data.shape[0], 1, Test_Data.shape[1]))
    model = Sequential()
    #model.add(Embedding(n_unique_words, 128, input_length=maxlen))
    model.add(LSTM(64, return_sequences=True))
    model.add(attention(return_sequences=True))  # receive 3D and output 3D
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    try :
        model.fit(trainX, Train_Target, epochs=5, batch_size=1, verbose=2)
    except:
        testPredict = model.predict(testX)
    eval =evaluation(testPredict.astype('int'), Test_Target)
    return eval

