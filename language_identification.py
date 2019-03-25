import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import os
import librosa as lr
import shutil
import dask.array as da
import h5py
import glob
# import models as mod
from sklearn.metrics import classification_report, confusion_matrix


from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, ConvLSTM2D, MaxPool1D, MaxPooling1D
from keras.layers import Dropout, Input, BatchNormalization, Bidirectional, Activation, ConvRNN2D, SimpleRNN,RNN, Conv1D,GRU
from keras.layers import LSTM, TimeDistributed, GlobalAveragePooling1D, Convolution2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import modls as mod

#from extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
import dask.array.image

class language_model:
    def __init__(self):
        self.data_size = 3008
        self.tr_size = 2105
        self.va_size = 601
        self.te_size = 302
        self.in_dim = (192, 192, 1)  # input image size
        self.out_dim = 8  # output langauge
        self.batch_size = 32
        self.audio_path = 'data/mp3/'

    def convertor_audio_image(self, path, height = 198, width = 198 ):
        """
        :param path:
        :param heigth: height for the image to be created, default value is 198
        :param width:  width for the image to be created, default value is 198
        :return: spectrogram image
        """
        wave, sampling_rate = lr.load(path, res_type='kaiser_fast')
        hl = wave.shape[0] // (width * 1.1)
        spectrogram = lr.feature.melspectrogram(wave, n_mels=height, hop_length=int(hl))
        # this is scale the frequency and intensity, to avoid any loss of information in lower
        #frequency while compressing higher frequencies
        spectrogram_log_image = lr.logamplitude(spectrogram)**2
        strt = (spectrogram_log_image.shape[1] - width) // 2
        return spectrogram_log_image[:, strt:strt+width]

    def audio_processor(self, input, ouput):
        """
        :param input: input location for mp3 files
        :param ouput: output location where images need to be stored
        """
        os.makedirs(ouput, exist_ok=True)
        files = glob.glob(input + '*.mp3')
        strt = len(input)

        for file in files:
            file = file.replace("\\", "//")
            print("file" + "" + file)
            image = self.convertor_audio_image(file)
            """
            saving all the log scaled spectrogram images in the same name 
            of the file name in the jpg format in the output folder
            """
            sp.misc.imsave(ouput + file[strt:] + '.jpg', image)

    def jpg_compression(self, input, output, name):
        """
        :param input: input location for mp3 files
        :param ouput: output location where images need to be stored
        :param name : location to store
        """
        da.image.imread(input + '*.jpg').to_hdf5(output, name)


    def loading_output(self):
        """
        This method loads the labels for the data
        """
        self.y = pd.read_csv('data/train_list.csv')['Language']
        self.y = pd.get_dummies(self.y)
        self.y = self.y.reindex_axis(sorted(self.y.columns), axis=1)
        self.y = self.y.values
        self.y = da.from_array(self.y, chunks=1000)


    def loading_input(self):
        """
        loading the features back from the dask array
        """
        self.x = h5py.File('data/data.h5')['data']
        self.x = da.from_array(self.x, chunks=1000)

    def loading_input_output(self):

        """
        This is shuffle and divide the data for input,
        test and validation
        """
        shfl = np.random.permutation(self.data_size)
        tr_idx = shfl[:self.tr_size]
        va_idx = shfl[self.tr_size:self.tr_size + self.va_size]
        te_idx = shfl[self.tr_size + self.va_size:]


        """
        loading back from dask array based on the size specified for training,
        validation and testing
        """

        self.x[tr_idx].to_hdf5('data/x_tr.h5', 'x_tr')
        self.y[tr_idx].to_hdf5('data/y_tr.h5', 'y_tr')
        self.x[va_idx].to_hdf5('data/x_va.h5', 'x_va')
        self.y[va_idx].to_hdf5('data/y_va.h5', 'y_va')
        self.x[te_idx].to_hdf5('data/x_te.h5', 'x_te')
        self.y[te_idx].to_hdf5('data/y_te.h5', 'y_te')


        self.x_tr = da.from_array(h5py.File('data/x_tr.h5')['x_tr'], chunks=1000)
        self.x_tr = self.x_tr[..., None]
        self.y_tr = da.from_array(h5py.File('data/y_tr.h5')['y_tr'], chunks=1000)

        self.x_va = da.from_array(h5py.File('data/x_va.h5')['x_va'], chunks=1000)
        self.y_va = da.from_array(h5py.File('data/y_va.h5')['y_va'], chunks=1000)
        self.x_va = self.x_va[..., None]


        self.x_te = da.from_array(h5py.File('data/x_te.h5')['x_te'], chunks=1000)
        self.y_te = da.from_array(h5py.File('data/y_te.h5')['y_te'], chunks=1000)
        self.x_te = self.x_te[..., None]

        self.x_tr /= 255.
        self.x_va /= 255.
        self.x_te /= 255.


    def LSTMmodel(self) :
        i = Input(shape= self.in_dim)

        model=Sequential();

        model.add(TimeDistributed(Conv1D(16,3,padding='same'),  input_shape=(192,192,1)))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(Conv1D(32, 3)))
        model.add(TimeDistributed(Activation('relu')))
        # model.add(TimeDistributed(Conv1D(64, 3)))
        # model.add(TimeDistributed(Activation('relu')))
        # model.add(TimeDistributed(Conv1D(128, 3)))
        # model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=(2))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Flatten()))
        return model

    def confusion(self, model):
        predict = model.predict(self.x_te)
        pred = np.zeros((predict.shape[0], predict.shape[1]))

        lis = list()
        for i in range(predict.shape[0]):
            best = -1
            for j in range(predict.shape[1]):
                if best < predict[i][j]:
                    best = predict[i][j]
                    index = j
            #    print(str(predict[i]) + '' + str(index))
            lis.append(index)
            pred[i][index] = int(1)

        lispred = list()
        for i in range(self.y_te.shape[0]):
            best = -1
            for j in range(self.y_te.shape[1]):
                if best < self.y_te[i][j]:
                    best = self.y_te[i][j]
                    index = j
            # print(str(y_te[i])+''+str(index))
            lispred.append(index)
            pred[i][index] = int(1)

        print(confusion_matrix(lispred, lis))

    def run(self):
        self.audio_processor('data/mp3/', 'data/jpg/')
        self.jpg_compression()
        self.loading_input()
        self.loading_output()
        self.loading_input_output()

        m = mod.GRUmodel(self.in_dim,self.out_dim)
        m.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        m.fit(self.x_tr, self.y_tr, epochs=1, verbose=1, validation_data=(self.x_va, self.y_va))

        m = mod.BGRU(self.in_dim, self.out_dim)
        m.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        m.fit(self.x_tr, self.y_tr, epochs=1, verbose=1, validation_data=(self.x_va, self.y_va))


        m = mod.CNN(self.in_dim, self.out_dim)
        m.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        m.fit(self.x_tr, self.y_tr, epochs=1, verbose=1, validation_data=(self.x_va, self.y_va))

        m = mod.DNN(self.in_dim, self.out_dim)
        m.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        m.fit(self.x_tr, self.y_tr, epochs=1, verbose=1, validation_data=(self.x_va, self.y_va))

a = language_model()
a.run()