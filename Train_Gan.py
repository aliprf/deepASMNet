from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from hg_Class import HourglassNet
from cnn_model import CNNModel
import tensorflow as tf
import keras
from skimage.transform import resize

from keras.regularizers import l2
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Deconvolution2D, Input

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


class TrainGan:
    def __init__(self, num_points=68):
        self.num_points = num_points

    def create_hm_generator_net(self):
        """img & hm -->  hm_p"""
        cnn = CNNModel()
        model = cnn.mn_asm_v1(None)

        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False),
                      metrics=['mse'])

        return model

    def create_reg_net(self, input_tensor, input_shape):
        """img & point ---> point_p"""
        cnn = CNNModel()
        model = cnn.mobileNet_v2_main_discriminator(input_tensor, input_shape)
        # model = cnn.create_shallow_reg(num_branches=self.num_points)
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False),
                      metrics=['accuracy'])
        return model

    def fuse_img(self, img, hm_p, batch_size):
        """

        :param img: the input image :       [224,224,3]
        :param hm_p: the predicted heatmap: [56,56,68]
        :return: fused img& hm:             [32,32,1]
        """
        f_img = np.ones(shape=[self.num_points, batch_size, 32, 32, 1])
        return f_img

    def hm_to_point(self, hm_model_out, batch_size):
        """

        :param hm_model_out: [batch_size, 56, 56, 68]
        :return: 68 * [batch_size, 32, 32]
        """
        result = tf.ones(shape=[batch_size, 136])
        # result = hm_model_out[:, :, :, :3]
        return result

    def get_batch_sample(self, bath_size):
        """

        :param bath_size:
        :return: images:[batch, 224,224,3] & hms:[batch, 56,56,68] & lbls:[batch, 2,1,68]
        """
        imgs = np.ones(shape=[bath_size, 224, 224, 3])
        hms = np.ones(shape=[bath_size, 56, 56, self.num_points])
        lbls = np.ones(shape=[bath_size, 2, 1, self.num_points])

        return imgs, hms, lbls

    def create_seq_model(self):
        epochs = 10
        steps_per_epoch = 100
        batch_size = 10

        hm_model = self.create_hm_generator_net()
        reg_model = self.create_reg_net(input_tensor=None, input_shape=[56, 56, 68])

        # reg_model.trainable = False

        seq_model_input = Input(shape=(224, 224, 3))
        hm_model_out = hm_model(seq_model_input)

        # hm_model_out_reshape = hm_model_out
        # hm_model_out_reshape = self.hm_to_point(hm_model_out, batch_size)
        # imp_1 = hm_model_out[0]

        seq_model_output = reg_model(hm_model_out)

        seq_model = Model(seq_model_input, outputs=seq_model_output)

        seq_model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer="sgd")

        seq_model.summary()
        model_json = seq_model.to_json()
        with open("seq_model.json", "w") as json_file:
            json_file.write(model_json)

        for epoch in range(epochs):
            for batch in range(steps_per_epoch):

                imgs, hms, lbls = self.get_batch_sample(batch_size)

                hm_ps = hm_model.predict_on_batch(imgs)

                x = np.concatenate((hms, hm_ps))

                disc_y = np.zeros(2 * batch_size)
                disc_y[:batch_size] = 0.9

                d_loss = reg_model.train_on_batch(x, disc_y)

                y_gen = np.ones(batch_size)
                g_loss = seq_model.train_on_batch(imgs, y_gen)

                print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')





