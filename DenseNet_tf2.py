from __future__ import absolute_import, division, print_function, unicode_literals
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D, PReLU, LeakyReLU
from tensorflow.keras.layers import Input, Flatten, add, Dense, Activation, Reshape
from tensorflow.keras.layers import Dropout, GlobalAveragePooling3D, concatenate, Softmax
from tensorflow.keras.layers import BatchNormalization, Concatenate, AveragePooling3D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
#from tensorflow.config.gpu import set_per_process_memory_growth
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K



gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)



def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)


def exp_decay(epoch):
    initial_lrate = 0.0001
    k = 0.04
    lrate = initial_lrate * math.exp(-k)
    return lrate

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

class DataFeedGenerator(tf.keras.utils.Sequence):

    def __init__(self,list_IDs,x1,x2,y,batch_size=32, dim=(44,52,52), n_channels=1, n_classes=1, shuffle=False, name="Training"):
        self.dim = dim
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.Y = y
        self.X1 = x1
        self.X2 = x2
        self.currentX1 = None
        self.currentX2 = None
        self.currentY = None
        self.batch_index = 0
        self.n_channels = n_channels
        self.classes = n_classes
        self.shuffle = shuffle
        self.name = name
        self.on_epoch_end()

    def __len__(self):
        n = math.ceil(self.X1.shape[0] / self.batch_size)
        print(self.name, "__len__", n)
        return n

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X1,X2,Y = self.__data_generation(list_IDs_temp)
        
        return [X1, X2], Y


    def __data_generation(self,list_IDs_temp):
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X1[i,] = self.X1[ID]
            X2[i,] = self.X2[ID]
            Y[i,] = self.Y[ID]
        return X1,X2,Y



hipp_L_train = HDF5Matrix('ADNI_data_t1.hdf5', 'hipp_L_train')
hipp_R_train = HDF5Matrix('ADNI_data_t1.hdf5', 'hipp_R_train')
y_train = HDF5Matrix('ADNI_data_t1.hdf5', 'y_train')

hipp_L_test = HDF5Matrix('ADNI_data_t1.hdf5', 'hipp_L_test')
hipp_R_test = HDF5Matrix('ADNI_data_t1.hdf5', 'hipp_R_test')
y_test = HDF5Matrix('ADNI_data_t1.hdf5', 'y_test')

def dense_block(x, blocks, name):

    for i in range(blocks):
        x = conv_block(x,16,name=name + '_block' + str(i + 1))

    return x

def transition_block(x, reduction, name):

    bn_axis = 4
    weight_decay = 1e-4
    eps = 1.1e-5
    x = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + '_bn')(x)
    x = LeakyReLU(alpha=0.3, name=name + '_relu')(x)
    x = Conv3D(int(x.shape[bn_axis] * reduction), kernel_size=(1,1,1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay), name=name + '_conv')(x)
    x = AveragePooling3D((2,2,2), strides=(2,2,2), name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):

    bn_axis = 4
    weight_decay = 1e-4
    eps = 1.1e-5
    x1 = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + '_0_bn')(x)
    x1 = LeakyReLU(alpha=0.3, name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, kernel_size=(1,1,1), padding='same', use_bias=False, name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + '_1_bn')(x1)
    x1 = LeakyReLU(alpha=0.3, name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, kernel_size=(3,3,3), padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x1 = Dropout(0.2, name=name + '_drop')(x1)
    x = Concatenate(axis=bn_axis, name = name + '_concat')([x, x1])
    return x 

def DenseNet(x, blocks, name):
    
    weight_decay = 1e-4
    eps = 1.1e-5
    #initial_layer
    x = Conv3D(64, kernel_size=(5,5,5), padding='same', use_bias=False, name=name + 'conv1/conv')(x)
    x = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + 'conv1/bn')(x)
    x = LeakyReLU(alpha=0.3, name=name + 'conv1/relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=2, name=name + 'pool1')(x)

    x = dense_block(x, blocks[0], name=name + 'conv2')
    x = transition_block(x, 0.5, name=name + 'pool2')
    x = dense_block(x, blocks[1], name=name + 'conv3')
    x = transition_block(x, 0.5, name=name + 'pool3')
    x = dense_block(x, blocks[2], name=name + 'conv4')
    x = transition_block(x, 0.5, name=name + 'pool4')
    x = dense_block(x, blocks[3], name=name + 'conv5')

    x = GlobalAveragePooling3D(name=name + 'avg_pool')(x)
    return x

epochs = 150
batch_size = 64
learningRate = 0.0001


initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = initial_learning_rate,
        decay_steps = 10000,
        decay_rate=0.90,
        staircase=True)
adam = Adam(learning_rate = lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
#sgd = SGD(lr=lr_schedule, momentum=0.9)
trainingGen = DataFeedGenerator(list_IDs=list(range(0,hipp_L_train.shape[0])), x1=hipp_L_train, x2=hipp_R_train, y=y_train, batch_size = batch_size, name="Training Gen")
validationGen = DataFeedGenerator(list_IDs=list(range(0,hipp_L_test.shape[0])), x1=hipp_L_test, x2=hipp_R_test, y=y_test, batch_size = batch_size, name="Validation Gen")

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():


    left_hipp = keras.Input(shape=(44,52,52,1), name='Input1')
    right_hipp = keras.Input(shape=(44,52,52,1), name='Input2')

    _L = DenseNet(left_hipp, [3,3,3,3], name='left_')
    _R = DenseNet(right_hipp, [3,3,3,3], name='right_')

    combined = concatenate([_L, _R])


    x = Dense(128, kernel_regularizer=l2(l=0.01))(combined)
    x = Activation('relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[left_hipp, right_hipp], outputs=output)


    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    checkpoint = ModelCheckpoint("weights-{epoch:02d}-{val_accuracy:.4f}.hdf5", monitor='val_accuracy', verbose=1, save_freq='epoch' ,save_best_only=False, mode='max')
model.summary()
history = model.fit_generator(generator = trainingGen, verbose=1, validation_data=validationGen, callbacks=[checkpoint], epochs = epochs, shuffle=False)

print(history.history.keys())

model.save('my_model.h5')




