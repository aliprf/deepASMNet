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
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, \
    Deconvolution2D, Input

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from clr_callback import CyclicLR
from datetime import datetime

import cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio


class CNNModel:
    def get_model(self, train_images, arch, num_output_layers):
        if arch == 'asmnet':
            model = self.create_asmnet(inp_shape=[224, 224, 3], num_branches=num_output_layers)
        elif arch == 'mb_mn':
            model = self.create_multi_branch_mn(inp_shape=[224, 224, 3], num_branches=num_output_layers)
            # model = cnn.create_multi_branch_mn_one_input(inp_shape=[224, 224, 3], num_branches=self.num_output_layers)
        elif arch == 'mn_asm_0':
            model = self.mn_asm_v0(train_images)
        elif arch == 'mn_asm_1':
            model = self.mn_asm_v1(train_images)
        elif arch == 'hg':
            model = self.hour_glass_network(num_stacks=num_output_layers)
        elif arch == 'mn_r':
            model = self.mnv2_hm(tensor=train_images)
        return model

    def hour_glass_network(self, num_classes=68, num_stacks=10, num_filters=256,
                           in_shape=(224, 224), out_shape=(56, 56)):
        hg_net = HourglassNet(num_classes=num_classes, num_stacks=num_stacks,
                              num_filters=num_filters,
                              in_shape=in_shape,
                              out_shape=out_shape)
        model = hg_net.build_model(mobile=True)
        return model

    def mn_asm_v0(self, tensor):
        """
            has only one output
            we use custom loss for this network and using ASM to correct points after that
        """

        # block_13_project_BN block_10_project_BN block_6_project_BN
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

        out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, out_heatmap)

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mn_asm_v0.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mn_asm_v1(self, tensor):
        # block_13_project_BN block_10_project_BN block_6_project_BN
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        # '''block_1 {  block_6_project_BN 14, 14, 46 '''
        # x = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 46
        # x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
        #                     name='block_1_deconv_1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        # x = BatchNormalization(name='block_1_out_bn_1')(x)
        #
        # x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
        #                     name='block_1_deconv_2', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        # x = BatchNormalization(name='block_1_out_bn_2')(x)
        #
        # block_1_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_1_out')(x)
        # '''block_1 }'''

        '''block_2 {  block_10_project_BN 14, 14, 96 '''
        x = mobilenet_model.get_layer('block_10_project_BN').output  # 14, 14, 96
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv_1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        x = BatchNormalization(name='block_2_out_bn_1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv_2', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        x = BatchNormalization(name='block_2_out_bn_2')(x)

        block_2_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_2_out')(x)
        '''block_2 }'''

        '''block_3 {  block_13_project_BN 7, 7, 160 '''
        x = mobilenet_model.get_layer('block_13_project_BN').output  # 7, 7, 160
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_1', kernel_initializer='he_uniform')(x)  # 14, 14, 128
        x = BatchNormalization(name='block_3_out_bn_1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_2', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        x = BatchNormalization(name='block_3_out_bn_2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_3', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        x = BatchNormalization(name='block_3_out_bn_3')(x)

        block_3_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_3_out')(x)

        '''block_3 }'''

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
            # block_1_out,  # 85
            # block_2_out,  # 90
            # block_3_out,  # 97
            out_heatmap  # 100
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mn_asm_v1.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    # def create_multi_branch_mn(self, inp_shape, num_branches):
    #     branches = []
    #     inputs = []
    #     for i in range(num_branches):
    #         inp_i, br_i = self.create_branch_mn(prefix=str(i), inp_shape=inp_shape)
    #         inputs.append(inp_i)
    #         branches.append(br_i)
    #
    #     revised_model = Model(inputs[0], branches[0], name='multiBranchMN')
    #     # revised_model = Model(inputs, branches, name='multiBranchMN')
    #
    #     revised_model.layers.pop(0)
    #
    #     new_input = Input(shape=inp_shape)
    #
    #     revised_model = Model(new_input, revised_model.outputs)
    #
    #     revised_model.summary()
    #
    #     model_json = revised_model.to_json()
    #     with open("MultiBranchMN.json", "w") as json_file:
    #         json_file.write(model_json)
    #     return revised_model
    #

    def create_multi_branch_mn(self, inp_shape, num_branches):

        mobilenet_model = mobilenet_v2.MobileNetV2_mb(3, input_shape=inp_shape,
                                        alpha=1.0,
                                        include_top=True,
                                        weights=None,
                                        input_tensor=None,
                                        pooling=None)

        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        outputs = []
        for i in range(num_branches):
            prefix = str(i)
            '''heatmap can not be generated from activation layers, so we use out_relu'''
            x = mobilenet_model.get_layer('out_relu'+prefix).output  # 7, 7, 1280

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
            x = BatchNormalization(name=prefix + 'out_bn1')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
            x = BatchNormalization(name=prefix + 'out_bn2')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
            x = BatchNormalization(name=prefix + 'out_bn3')(x)

            out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)
            outputs.append(out_heatmap)

        revised_model = Model(inp, outputs)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("MultiBranchMN.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def create_branch_mn(self, prefix, inp_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)
        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name=prefix + 'out_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name=prefix + 'out_bn2')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name=prefix + 'out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)

        for layer in mobilenet_model.layers:
            layer.name = layer.name + '_' + prefix
        return inp, out_heatmap

    # def create_multi_branch_mn_one_input(self, inp_shape, num_branches):
    #     mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
    #                                                alpha=1.0,
    #                                                include_top=True,
    #                                                weights=None,
    #                                                input_tensor=None,
    #                                                pooling=None)
    #     mobilenet_model.layers.pop()
    #     inp = mobilenet_model.input
    #     outputs = []
    #     relu_name = 'out_relu'
    #     for i in range(num_branches):
    #         x = mobilenet_model.get_layer(relu_name).output  # 7, 7, 1280
    #         prefix = str(i)
    #         for layer in mobilenet_model.layers:
    #             layer.name = layer.name + prefix
    #
    #         relu_name = relu_name + prefix
    #
    #         '''heatmap can not be generated from activation layers, so we use out_relu'''
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
    #         x = BatchNormalization(name=prefix + 'out_bn1')(x)
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
    #         x = BatchNormalization(name=prefix +'out_bn2')(x)
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
    #         x = BatchNormalization(name=prefix+'out_bn3')(x)
    #
    #         out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix+'_out_hm')(x)
    #         outputs.append(out_heatmap)
    #
    #     revised_model = Model(inp, outputs)
    #
    #     revised_model.summary()
    #
    #     model_json = revised_model.to_json()
    #     with open("MultiBranchMN.json", "w") as json_file:
    #         json_file.write(model_json)
    #     return revised_model

    def create_asmnet(self, inp_shape, num_branches):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)

        mobilenet_model.layers.pop()
        inp = mobilenet_model.input
        outputs = []
        relu_name = 'out_relu'
        for i in range(num_branches):
            x = mobilenet_model.get_layer(relu_name).output  # 7, 7, 1280
            prefix = str(i)
            for layer in mobilenet_model.layers:
                layer.name = layer.name + prefix

            relu_name = relu_name + prefix

            '''heatmap can not be generated from activation layers, so we use out_relu'''

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix+'_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
            x = BatchNormalization(name=prefix + 'out_bn1')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix+'_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
            x = BatchNormalization(name=prefix +'out_bn2')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix+'_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
            x = BatchNormalization(name=prefix+'out_bn3')(x)

            out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix+'_out_hm')(x)
            outputs.append(out_heatmap)

        revised_model = Model(inp, outputs)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("asmnet.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

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
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
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

    def mobileNet_v2_main_discriminator(self, tensor, input_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_2').output  # 1280
        softmax = Dense(1, activation='sigmoid', name='out')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, softmax)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
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
