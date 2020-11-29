from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig, CofwConf, WflwConf
from tf_record_utility import TFRecordUtility
from clr_callback import CyclicLR
from cnn_model import CNNModel
from custom_Losses import Custom_losses
from Data_custom_generator import CustomHeatmapGenerator
from PW_Data_custom_generator import PWCustomHeatmapGenerator
import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path
from keras import losses
from keras import backend as K
import csv
from skimage.io import imread
import pickle


class StudentTrainer:

    def __init__(self, dataset_name, arch):
        self.dataset_name = dataset_name

        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.number_of_all_sample
            self.output_len = IbugConf.num_of_landmarks * 2
            self.tf_train_path = IbugConf.augmented_train_tf_path + 'train100.tfrecords'
            self.tf_eval_path = IbugConf.augmented_train_tf_path + 'eval100.tfrecords'
            self.img_path = IbugConf.augmented_train_image

        elif dataset_name == DatasetName.cofw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.number_of_all_sample
            self.output_len = CofwConf.num_of_landmarks * 2
            self.tf_train_path = CofwConf.augmented_train_tf_path + 'train100.tfrecords'
            self.tf_eval_path = CofwConf.augmented_train_tf_path + 'eval100.tfrecords'
            self.img_path = CofwConf.augmented_train_image

        elif dataset_name == DatasetName.wflw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.number_of_all_sample
            self.output_len = WflwConf.num_of_landmarks * 2
            self.tf_train_path = WflwConf.augmented_train_tf_path + 'train100.tfrecords'
            self.tf_eval_path = WflwConf.augmented_train_tf_path + 'eval100.tfrecords'
            self.img_path = WflwConf.train_images_dir

        self.BATCH_SIZE = LearningConfig.batch_size
        self.STEPS_PER_VALIDATION_EPOCH = LearningConfig.steps_per_validation_epochs
        self.STEPS_PER_EPOCH = self.SUM_OF_ALL_TRAIN_SAMPLES // self.BATCH_SIZE
        self.EPOCHS = LearningConfig.epochs

        self.arch = arch

    def _create_generators(self):
        tf_utils = TFRecordUtility(self.output_len)

        filenames, labels = tf_utils.create_image_and_labels_name()
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
            filenames_shuffled, y_labels_shuffled, test_size=0.05, random_state=100, shuffle=True)

        save('x_train_filenames.npy', x_train_filenames)
        save('x_val_filenames.npy', x_val_filenames)
        save('y_train_filenames.npy', y_train)
        save('y_val_filenames.npy', y_val)

        return x_train_filenames, x_val_filenames, y_train, y_val

    def train(self, teachers_arch, teachers_weight_files, teachers_weight_loss,
              teachers_tf_train_paths, student_weight_file):
        """
        :param teachers_arch: an array containing architecture of teacher networks
        :param teachers_weight_files: an array containing teachers h5 files
        :param teachers_weight_loss: an array containing weight of teachers model in loss function
        :param teachers_tf_train_paths: an array containing path of train tf records
        :param student_weight_file : student h5 weight path
        :return: null
        """

        tf_record_util = TFRecordUtility(self.output_len)
        c_loss = Custom_losses()

        '''-------------------------------------'''
        '''     preparing student models        '''
        '''-------------------------------------'''
        teacher_models = []
        cnn = CNNModel()
        for i in range(len(teachers_arch)):
            student_train_images, student_train_landmarks = tf_record_util.create_training_tensor_points(
                tfrecord_filename=teachers_tf_train_paths[i], batch_size=self.BATCH_SIZE)
            model = cnn.get_model(train_images=student_train_images, arch=teachers_arch[i], num_output_layers=1,
                                  output_len=self.output_len, input_tensor=None)

            model.load_weights(teachers_weight_files[i])
            teacher_models.append(model)

        '''---------------------------------'''
        '''     creating student model      '''
        '''---------------------------------'''
        '''retrieve tf data'''
        train_images, train_landmarks = tf_record_util.create_training_tensor_points(
            tfrecord_filename=self.tf_train_path, batch_size=self.BATCH_SIZE)
        validation_images, validation_landmarks = tf_record_util.create_training_tensor_points(
            tfrecord_filename=self.tf_eval_path, batch_size=self.BATCH_SIZE)

        '''create model'''
        student_model = cnn.get_model(train_images=train_images, arch=self.arch, num_output_layers=1,
                                      output_len=self.output_len, input_tensor=train_images, inp_shape=None)
        if student_weight_file is not None:
            student_model.load_weights(student_weight_file)

        '''prepare callbacks'''
        callbacks_list = self._prepare_callback()

        ''' define optimizers'''
        optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)

        '''create loss'''
        # file = open("map_aug" + self.dataset_name, 'rb')
        file = open("map_orig" + self.dataset_name, 'rb')
        landmark_img_map = pickle.load(file)
        file.close()

        # loss_func = c_loss.custom_teacher_student_loss_cos(img_path=self.img_path, lnd_img_map=landmark_img_map,
        #                                                    teacher_models=teacher_models,
        #                                                    teachers_weight_loss=teachers_weight_loss,
        #                                                    bath_size=self.BATCH_SIZE,
        #                                                    num_points=self.output_len, cos_weight=cos_weight)

        loss_func = c_loss.custom_teacher_student_loss(img_path=self.img_path, lnd_img_map=landmark_img_map,
                                                       teacher_models=teacher_models,
                                                       teachers_weight_loss=teachers_weight_loss,
                                                       bath_size=self.BATCH_SIZE,
                                                       num_points=self.output_len,
                                                       ds_name=self.dataset_name, loss_type=0)

        '''compiling model'''
        student_model.compile(loss=loss_func,
                              optimizer=optimizer,
                              metrics=['mse', 'mae'],
                              target_tensors=train_landmarks)

        print('< ========== Start Training Student============= >')
        history = student_model.fit(train_images,
                                    train_landmarks,
                                    epochs=self.EPOCHS,
                                    steps_per_epoch=self.STEPS_PER_EPOCH,
                                    validation_data=(validation_images, validation_landmarks),
                                    validation_steps=self.STEPS_PER_VALIDATION_EPOCH,
                                    verbose=1, callbacks=callbacks_list)

    def _prepare_callback(self):
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=500, verbose=1, mode='min')
        file_path = "weights-{epoch:02d}-{loss:.5f}.h5"
        checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger('log.csv', append=True, separator=';')

        clr = CyclicLR(
            mode=LearningConfig.CLR_METHOD,
            base_lr=LearningConfig.MIN_LR,
            max_lr=LearningConfig.MAX_LR,
            step_size=LearningConfig.STEP_SIZE * (self.SUM_OF_ALL_TRAIN_SAMPLES // self.BATCH_SIZE))

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        return [checkpoint, early_stop, csv_logger, clr, tensorboard_callback]
