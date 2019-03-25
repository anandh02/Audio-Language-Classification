from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, ConvLSTM2D, MaxPool1D, MaxPooling1D
from keras.layers import Dropout, Input, BatchNormalization, Bidirectional, Activation, ConvRNN2D, SimpleRNN,RNN, Conv1D
from keras.optimizers import Nadam
from keras.layers import LSTM, TimeDistributed, GlobalAveragePooling1D, Convolution2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, ConvLSTM2D, MaxPool1D, MaxPooling1D, GRU,MaxPooling1D

from keras.utils import np_utils

# class modls:
#     def __init__(self,in_dim,output_dim):
#         self.in_dim = 8
#         self.output_dim =34

def CNN(in_dim,output_dim) :
    i = Input(shape=in_dim)
    m = Conv2D(16, (3, 3), activation='elu', padding='same')(i)
    m = MaxPooling2D()(m)
    m = Conv2D(32, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Conv2D(64, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Conv2D(128, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Conv2D(256, (3, 3), activation='elu', padding='same')(m)
    m = MaxPooling2D()(m)
    m = Flatten()(m)
    m = Dense(512, activation='elu')(m)
    m = Dropout(0.5)(m)
    o = Dense(output_dim, activation='softmax')(m)

    model = Model(inputs=i, outputs=o)
    model.summary()
    return model

def GRUmodel(in_dim, output_dim, layers = True):
    i = Input(shape= in_dim)

    model=Sequential();
    model.add(TimeDistributed(Conv1D(16, 3, padding='same'), input_shape=(192, 192, 1)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(32, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(64, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(128, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(256, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(512, 3)))
    model.add(TimeDistributed(Activation('relu')))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=(2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    # model.add(Reshape((192,192)))
    # model.add(Bidirectional(GRU(16, return_sequences=True, name="lstm_layer")));
    if layers:
        model.add((GRU(16, return_sequences=True, name="GRU_layer1")));
        model.add((GRU(8, return_sequences=False, name="GRU_layer2")));
    else:
        model.add((GRU(output_dim, return_sequences=False, name="GRU_layer2")));
    return model

def BGRU(in_dim, output_dim, layers = False):

    i = Input(shape= in_dim)

    model=Sequential();
    model.add(TimeDistributed(Conv1D(16, 3, padding='same'), input_shape=(192, 192, 1)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(32, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(64, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(128, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(256, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv1D(512, 3)))
    model.add(TimeDistributed(Activation('relu')))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=(2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    # model.add(Reshape((192,192)))
    # model.add(Bidirectional(GRU(16, return_sequences=True, name="lstm_layer")));
    if (layers ):
        model.add(Bidirectional(GRU(16, return_sequences=True, name="lstm_layer1")));
        model.add(Bidirectional(GRU(4, return_sequences=False, name="lstm_layer2")));
    else:
        model.add(Bidirectional(GRU(4, return_sequences=False, name="BGRU_layer1")));
    return model

def DNN(in_dim,output_dim):

    i = Input(shape=in_dim)
    #print(in_dim)
    model = Dense(64, activation='relu')(i)
    model = Dense(128, activation='relu')(model)
    model = Dense(256, activation='relu')(model)
    model = Dense(512, activation='relu')(model)
    # model = Dense(780, activation='relu')(model)
    model = Flatten()(model)
    # model = Dense(255, activation='elu')(f)
    model = Dropout(0.5)(model)
    model = Dense(output_dim, activation='softmax')(model)
    # model.summary()
    Fodel = Model(inputs=i, outputs=model)

    return Fodel