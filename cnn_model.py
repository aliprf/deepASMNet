from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from hg_Class import HourglassNet

import tensorflow as tf
import keras
from skimage.transform import resize

from keras.regularizers import l2
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Deconvolution2D

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from clr_callback import CyclicLR
from datetime import datetime

import  cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio


class CNNModel:

    def hour_glass_network(self, num_classes=68, num_stacks=4, num_filters=512,
                           in_shape=(224, 224), out_shape=(56, 56)):
        hg_net = HourglassNet(num_classes=num_classes, num_stacks=num_stacks,
                              num_filters=num_filters,
                              in_shape=in_shape,
                              out_shape=out_shape)
        model = hg_net.build_model()
        return model

    def mnv2_hm(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input
        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mobileNet_v2_main(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu',
                  kernel_initializer='he_uniform')(x)
        Logits = Dense(LearningConfig.landmark_len, name='out')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, Logits)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

