import math

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from PIL import Image
from tensorflow.keras import backend as K
from scipy.spatial import distance

from cnn_model import CNNModel
from configuration import DatasetName, IbugConf, LearningConfig
from image_utility import ImageUtility
from pca_utility import PCAUtility
from tf_record_utility import TFRecordUtility

print(tf.__version__)


class Custom_losses:
    def custom_face_web_loss(self, bath_size, ds_name, num_points, loss_type, main_loss_wight,
                             inter_faceweb_weight, intra_faceweb_weight):
        def loss(y_true, y_pred):
            """"""
            '''calculate the main loss'''
            if loss_type == 0:
                '''MAE'''
                main_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            elif loss_type == 1:
                '''MSE'''
                main_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            '''calculate the inter faceweb distance: the distance between each facial elements(nose to mouth)'''
            inter_fb_gt = self._create_inter_fwd(ds_name, y_true, bath_size)
            inter_fb_pr = self._create_inter_fwd(ds_name, y_pred, bath_size)
            '''calculate the intra faceweb distance: the internal distance between a facial element(eye)'''

        return loss

    # def kd_loss(self, y_pr, y_gt, y_togh_t, y_tol_t, l_w_stu_t, l_w_togh_t, l_w_tol_t, loss_type=0):
    #     """"""
    #     '''calculate the sign of difference'''
    #     sign_delta_gt_and_tough_teacher = tf.sign((y_gt - y_togh_t))
    #     sign_delta_gt_and_tol_teacher = tf.sign((y_gt - y_tol_t))
    #     '''create weight_map'''
    #     weight_map_t_tough = tf.math.multiply(tf.ones_like(y_gt), l_w_togh_t)
    #     weight_map_t_tol = tf.math.multiply(tf.ones_like(y_gt), l_w_tol_t)
    #
    #     '''find indices that need to be modified'''
    #     minus_one_tough = tf.constant(l_w_togh_t, dtype=tf.float32)
    #     minus_one_tol = tf.constant(l_w_tol_t, dtype=tf.float32)
    #     where_cond_tough = tf.not_equal(sign_delta_gt_and_tough_teacher, minus_one_tough)
    #     where_cond_tol = tf.not_equal(sign_delta_gt_and_tol_teacher, minus_one_tol)
    #
    #     '''calculate the opposite sign items and fill map'''
    #     indices_tough = tf.where(where_cond_tough)
    #     indices_tol = tf.where(where_cond_tol)
    #     weight_map_t_tough = self._fill_opposite_sign_map(indices=indices_tough, y_gt=y_gt, y_pr=y_pr,
    #                                                       y_t=y_togh_t,
    #                                                       loss_weight=l_w_togh_t,
    #                                                       loss_map=weight_map_t_tough)
    #     weight_map_t_tol = self._fill_opposite_sign_map(indices=indices_tough, y_gt=y_gt, y_pr=y_pr,
    #                                                     y_t=y_tol_t,
    #                                                     loss_weight=l_w_tol_t,
    #                                                     loss_map=weight_map_t_tol)
    #     '''apply map to the teacher weights'''
    #     y_togh_t = tf.math.multiply(weight_map_t_tough, y_togh_t)
    #     y_tol_t = tf.math.multiply(weight_map_t_tol, y_tol_t)
    #
    #     return loss

    def kd_loss(self, x_pr, x_gt, x_tough, x_tol,
                alpha_tough, alpha_mi_tough,
                alpha_tol, alpha_mi_tol,
                main_loss_weight, tough_loss_weight, tol_loss_weight,
                num_of_landmarks):
        """"""
        '''creating np version of input tensors'''
        loss_shape = (x_pr.shape[0], x_pr.shape[1])
        np_x_pr = np.array(x_pr).reshape(x_pr.shape[0] * x_pr.shape[1])
        np_x_gt = np.array(x_gt).reshape(x_gt.shape[0] * x_gt.shape[1])
        np_x_tough = np.array(x_tough).reshape(x_tough.shape[0] * x_tough.shape[1])
        np_x_tol = np.array(x_tol).reshape(x_tol.shape[0] * x_tol.shape[1])
        # np_x_pr = K.eval(x_pr).reshape(x_pr.shape[0] * x_pr.shape[1])
        # np_x_gt = K.eval(x_gt).reshape(x_gt.shape[0] * x_gt.shape[1])
        # np_x_tough = K.eval(x_tough).reshape(x_tough.shape[0] * x_tough.shape[1])
        # np_x_tol = K.eval(x_tol).reshape(x_tol.shape[0] * x_tol.shape[1])

        '''calculate the weight map'''
        weight_map_tough = np.zeros_like(np_x_tough)
        weight_map_tol = np.zeros_like(np_x_tol)
        vv = np_x_pr.shape[0]
        for i in range(np_x_pr.shape[0]):
            weight_map_tough[i] = self.calc_teacher_weight_loss(x_pr=np_x_pr[i], x_gt=np_x_gt[i], x_t=np_x_tough[i],
                                                                alpha=alpha_tough, alpha_mi=alpha_mi_tough)
            weight_map_tol[i] = self.calc_teacher_weight_loss(x_pr=np_x_pr[i], x_gt=np_x_gt[i], x_t=np_x_tol[i],
                                                              alpha=alpha_tol, alpha_mi=alpha_mi_tol)
        '''reshape loss'''
        weight_map_tough = weight_map_tough.reshape(loss_shape)
        weight_map_tol = weight_map_tol.reshape(loss_shape)

        loss_tough = tf.reduce_mean(weight_map_tough * tf.abs(x_tough - x_pr))
        loss_tol = tf.reduce_mean(weight_map_tol * tf.abs(x_tol - x_pr))
        '''calculate the losses'''

        loss_main = main_loss_weight * tf.reduce_mean(tf.abs(x_gt - x_pr))
        loss_tough = tough_loss_weight * loss_tough
        loss_tol = tol_loss_weight * loss_tol
        loss_total = loss_main + loss_tough + loss_tol
        '''returns all losses'''
        return loss_total, loss_main, loss_tough, loss_tol

    def calc_teacher_weight_loss(self, x_pr, x_gt, x_t, alpha, alpha_mi):
        weight_loss_t = 0
        '''calculate betas'''
        beta = x_gt + 0.4 * abs(x_gt - x_t)
        beta_mi = x_gt - 0.4 * abs(x_gt - x_t)
        if x_t > x_gt:
            if x_pr >= x_t:
                weight_loss_t = alpha
            elif beta <= x_pr < x_t:
                weight_loss_t = alpha_mi
            elif x_gt <= x_pr < beta:
                weight_loss_t = (alpha_mi / (beta - x_gt)) * (x_pr - x_gt)
            elif beta_mi < x_pr < x_gt:
                weight_loss_t = (alpha / (beta_mi - x_gt)) * (x_pr - x_gt)
            elif x_pr <= beta_mi:
                weight_loss_t = alpha
        elif x_t < x_gt:
            if x_pr <= x_t:
                weight_loss_t = alpha
            elif x_t < x_pr <= beta_mi:
                weight_loss_t = alpha_mi
            elif beta_mi < x_pr <= x_gt:
                weight_loss_t = (-alpha_mi / (x_gt - beta_mi)) * (x_pr - x_gt)
            elif x_gt < x_pr <= beta:
                weight_loss_t = (alpha / (beta - x_gt)) * (x_pr - x_gt)
            elif x_pr > beta:
                weight_loss_t = alpha
        return weight_loss_t

    def custom_teacher_student_loss(self, lnd_img_map, img_path, teacher_models, teachers_weight_loss, bath_size,
                                    num_points, ds_name, loss_type):
        def loss(y_true, y_pred):
            image_utility = ImageUtility()

            t0_model = teacher_models[0]
            t1_model = teacher_models[1]
            l0_weight = teachers_weight_loss[0]
            l1_weight = teachers_weight_loss[1]

            y_true_n = tf.reshape(y_true, [bath_size, num_points], name=None)
            imgs_address = self.get_y(y_true_n, lnd_img_map, img_path)
            imgs_batch = [np.array(Image.open(img_file)) / 255.0 for img_file in imgs_address]

            y_pred_T0 = np.array([t0_model.predict(np.expand_dims(img, axis=0))[0] for img in imgs_batch])
            y_pred_T1 = np.array([t1_model.predict(np.expand_dims(img, axis=0))[0] for img in imgs_batch])

            '''test teacher Nets'''
            # counter = 0
            # for pre_points in y_pred_T1:
            #     labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = \
            #         image_utility.create_landmarks_from_normalized(pre_points, 224, 224, 112, 112)
            #     imgpr.print_image_arr((counter + 1) * 1000, imgs_batch[counter], landmark_arr_x_p, landmark_arr_y_p)
            #     counter += 1

            y_pred_Tough_ten = K.variable(y_pred_T0)
            y_pred_Tol_ten = K.variable(y_pred_T1)

            '''calculate the sign of difference'''
            sign_delta_gt_and_tough_teacher = tf.sign((y_true - y_pred_Tough_ten))
            sign_delta_gt_and_tol_teacher = tf.sign((y_true - y_pred_Tol_ten))
            '''for each point, if signs are the same, we sum losses, 
                    but if signs are different, we minus the 
                    teacher loses from the main loss, IF:
                    asb(y_pred - y_pred_tech_i)) < '''

            '''assign weight loss'''
            tough_teacher_weight_loss = 1
            tol_teacher_weight_loss = 1

            '''create weight_map'''
            weight_map_t_tough = tf.math.multiply(tf.ones_like(y_true), tough_teacher_weight_loss)
            weight_map_t_tol = tf.math.multiply(tf.ones_like(y_true), tol_teacher_weight_loss)

            '''find indices that need to be modified'''
            minus_one_tough = tf.constant(tough_teacher_weight_loss, dtype=tf.float32)
            minus_one_tol = tf.constant(tough_teacher_weight_loss, dtype=tf.float32)
            where_cond_tough = tf.not_equal(sign_delta_gt_and_tough_teacher, minus_one_tough)
            where_cond_tol = tf.not_equal(sign_delta_gt_and_tol_teacher, minus_one_tol)

            '''calculate the opposite sign items and fill map'''
            indices_tough = tf.where(where_cond_tough)
            indices_tol = tf.where(where_cond_tol)

            weight_map_t_tough = self._fill_opposite_sign_map(indices=indices_tough, y_gt=y_true, y_pr=y_pred,
                                                              y_t=y_pred_Tough_ten,
                                                              loss_weight=tough_teacher_weight_loss,
                                                              loss_map=weight_map_t_tough)
            weight_map_t_tol = self._fill_opposite_sign_map(indices=indices_tough, y_gt=y_true, y_pr=y_pred,
                                                            y_t=y_pred_Tol_ten,
                                                            loss_weight=tol_teacher_weight_loss,
                                                            loss_map=weight_map_t_tol)
            '''apply map to the teacher weights'''
            y_pred_Tough_ten = tf.math.multiply(weight_map_t_tough, y_pred_Tough_ten)
            y_pred_Tol_ten = tf.math.multiply(weight_map_t_tol, y_pred_Tol_ten)
            ''''''
            if loss_type == 0:
                '''MAE'''
                mse_te0 = tf.reduce_mean(tf.abs(y_pred - y_pred_Tough_ten))
                mse_te1 = tf.reduce_mean(tf.abs(y_pred - y_pred_Tol_ten))
                mse_main = tf.reduce_mean(tf.abs(y_pred - y_true))
            elif loss_type == 1:
                '''MSE'''
                mse_te0 = tf.reduce_mean(tf.square(y_pred - y_pred_Tough_ten))
                mse_te1 = tf.reduce_mean(tf.square(y_pred - y_pred_Tol_ten))
                mse_main = tf.reduce_mean(tf.square(y_pred - y_true))
                '''     or:'''
                # mse = tf.keras.losses.MeanSquaredError()
                # mse_te0 = mse(y_pred_T0_ten, y_true)
                '''     or:'''
                # mse_main = K.mean(K.square(y_pred - y_true))
                # mse_main = K.mean(K.square(y_pred - y_true))

            return 10 * mse_main + ((l0_weight * mse_te0) + (l1_weight * mse_te1))

        return loss

    def _fill_opposite_sign_map(self, indices, y_gt, y_pr, y_t, loss_weight, loss_map):
        for index in indices:
            loss_map[index] = loss_weight * y_pr[index] / abs(y_gt[index] - y_t[index])
        return loss_map

    def get_y(self, y_true_n, lnd_img_map, img_path):
        vec_mse = K.eval(y_true_n)
        print(vec_mse.shape)
        imgs = []
        for lnd in vec_mse:
            print(lnd.shape)
            # lnd = lnd.tostring()
            # print(lnd)
            # lnd_hash = self.get_hash_key(lnd)
            lnd_hash = self.np_to_str(lnd)
            print("-------------------")
            print(lnd_hash)
            print("-------------------")
            key = lnd_hash
            img_name = lnd_img_map[key]
            imgs.append(img_path + img_name)
        return np.array(imgs)

    def np_to_str(self, input):
        str_out = ''
        for item in input:
            str_out += str(item)[:3]
        return str_out

    def get_hash_key(self, input):
        return str(hash(str(input).replace("\n", "").replace(" ", "")))

    def _decode_tf_file_name(self, file_name):
        return str(file_name).replace("X", "")

    # def _create_inter_fwd(self, ds_name, y_pred, bath_size):
    #     """based on the database, we return the distance between each facial elements"""
    #     if ds_name == DatasetName.cofw:
    #     # elif ds_name == DatasetName.ibug:
    #     # elif ds_name == DatasetName.wflw:

    def asm_assisted_loss(self, hmp_85, hmp_90, hmp_95):
        def loss(y_true, y_pred):
            return K.mean(K.square(y_pred - y_true))

        return loss

    def _calculate_mse(self, y_p, y_t):
        mse = (np.square(y_p - y_t)).mean(axis=None)
        # print('y_p: '+str(y_p.shape))
        # print('mse: '+str(mse.shape))
        # loss = 0
        # for j in range(len(y_p)):
        #     loss += (y_p[j] - y_t[j]) ** 2
        # loss /= len(y_p)

        # print('calculate_mse: ' + str(mse))
        return mse

    def _generate_distance_matrix(self, xy_arr):
        x_arr = xy_arr[[slice(None, None, 2) for _ in range(xy_arr.ndim)]]
        y_arr = xy_arr[[slice(1, None, 2) for _ in range(xy_arr.ndim)]]

        d_matrix = np.zeros(shape=[len(x_arr), len(y_arr)])
        for i in range(0, x_arr.shape[0], 1):
            for j in range(i + 1, x_arr.shape[0], 1):
                p1 = [x_arr[i], y_arr[i]]
                p2 = [x_arr[j], y_arr[j]]
                d_matrix[i, j] = distance.euclidean(p1, p2)
                d_matrix[j, i] = distance.euclidean(p1, p2)
        return d_matrix

    def _depart_facial_point(self, xy_arr):
        face = xy_arr[0:54]  # landmark_face_len = 54
        nose = xy_arr[54:72]  # landmark_nose_len = 18
        leys = xy_arr[72:84]  # landmark_eys_len = 24
        reys = xy_arr[84:96]  # landmark_eys_len = 24
        mouth = xy_arr[96:136]  # landmark_mouth_len = 40
        return face, nose, leys, reys, mouth

    def custom_loss_hm(self, ten_hm_t, ten_hm_p):
        # print(ten_hm_t.get_shape())  #  [None, 56, 56, 68]
        # print(ten_hm_p.get_shape())

        tf_utility = TFRecordUtility()

        sqr = K.square(ten_hm_t - ten_hm_p)  # [None, 56, 56, 68]
        mean1 = K.mean(sqr, axis=1)
        mean2 = K.mean(mean1, axis=1)
        tensor_mean_square_error = K.mean(mean2, axis=1)

        # print(tensor_mean_square_error.get_shape().as_list())  # [None, 68]

        # vec_mse = K.eval(tensor_mean_square_error)
        # print("mse.shape:")
        # print(vec_mse.shape)  # (50, 68)
        # print(vec_mse)
        # print("----------->>>")

        '''calculate points from generated hm'''

        p_points_batch = tf.stack([tf_utility.from_heatmap_to_point_tensor(ten_hm_p[i], 5, 1)
                                   for i in range(LearningConfig.batch_size)])

        t_points_batch = tf.stack([tf_utility.from_heatmap_to_point_tensor(ten_hm_t[i], 5, 1)
                                   for i in range(LearningConfig.batch_size)])

        '''p_points_batch is [batch, 2, 68]'''
        sqr_2 = K.square(t_points_batch - p_points_batch)  # [None, 2, 68]
        mean_1 = K.mean(sqr_2, axis=1)
        tensor_indices_mean_square_error = K.mean(mean_1, axis=1)

        # tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, tensor_indices_mean_square_error])

        tensor_total_loss = tf.add(tensor_mean_square_error, tensor_indices_mean_square_error)
        return tensor_total_loss

    def custom_loss_hm_distance(self, ten_hm_t, ten_hm_p):
        print(ten_hm_t.get_shape().as_list())  # [None, 56, 56, 68]
        print(ten_hm_p.get_shape())

        tf_utility = TFRecordUtility()

        sqr = K.square(ten_hm_t - ten_hm_p)  # [None, 56, 56, 68]
        mean1 = K.mean(sqr, axis=1)
        mean2 = K.mean(mean1, axis=1)
        tensor_mean_square_error = K.mean(mean2, axis=1)
        # print(tensor_mean_square_error.get_shape().as_list())  # [None, 68]

        # vec_mse = K.eval(tensor_mean_square_error)
        # print("mse.shape:")
        # print(vec_mse.shape)  # (50, 68)
        # print(vec_mse)
        # print("----------->>>")

        '''convert tensor to vector'''
        vec_hm_p = K.eval(ten_hm_p)
        vec_hm_t = K.eval(ten_hm_t)

        loss_array = []

        for i in range(LearningConfig.batch_size):
            '''convert heatmap to points'''
            x_h_p, y_h_p, xy_h_p = tf_utility.from_heatmap_to_point(vec_hm_p[i], 5, 1)
            x_h_t, y_h_t, xy_h_t = tf_utility.from_heatmap_to_point(vec_hm_t[i], 5, 1)

            '''normalise points to be in [0, 1]'''
            x_h_p = x_h_p / 56
            y_h_p = y_h_p / 56
            xy_h_p = xy_h_p / 56
            x_h_t = x_h_t / 56
            y_h_t = y_h_t / 56
            xy_h_t = xy_h_t / 56
            '''test print images'''
            # imgpr.print_image_arr(i + 1, np.zeros(shape=[56, 56]), x_h_t, y_h_t)
            # imgpr.print_image_arr((i + 1)*1000, np.zeros(shape=[56, 56]), x_h_p, y_h_p)

            # print('--xy_h_p:---')
            # print(xy_h_p)
            # print('--xy_h_t:---')
            # print(xy_h_t)

            face_p, mouth_p, nose_p, leye_p, reye_p = self._depart_facial_point(xy_h_p)
            face_t, mouth_t, nose_t, leye_t, reye_t = self._depart_facial_point(xy_h_t)

            '''generate facial distance matrix'''
            face_p_mat, face_t_mat = self._generate_distance_matrix(face_p), self._generate_distance_matrix(face_t)
            mouth_p_mat, mouth_t_mat = self._generate_distance_matrix(mouth_p), self._generate_distance_matrix(mouth_t)
            nose_p_mat, nose_t_mat = self._generate_distance_matrix(nose_p), self._generate_distance_matrix(nose_t)
            leye_p_mat, leye_t_mat = self._generate_distance_matrix(leye_p), self._generate_distance_matrix(leye_t)
            reye_p_mat, reye_t_mat = self._generate_distance_matrix(reye_p), self._generate_distance_matrix(reye_t)

            '''calculate loss from each pair matrices'''

            face_loss = LearningConfig.reg_term_face * self._calculate_mse(face_p_mat, face_t_mat) / len(face_p)
            mouth_loss = LearningConfig.reg_term_mouth * self._calculate_mse(mouth_p_mat, mouth_t_mat) / len(mouth_p)
            nose_loss = LearningConfig.reg_term_nose * self._calculate_mse(nose_p_mat, nose_t_mat) / len(nose_p)
            leye_loss = LearningConfig.reg_term_leye * self._calculate_mse(leye_p_mat, leye_t_mat) / len(leye_p)
            reye_loss = LearningConfig.reg_term_reye * self._calculate_mse(reye_p_mat, reye_t_mat) / len(reye_p)

            loss_array.append(face_loss + mouth_loss + nose_loss + leye_loss + reye_loss)

            # print('mse[i]: ' + str(vec_mse[i]))
            # print('face_loss[i]: ' + str(face_loss))
            # print('mouth_loss[i]: ' + str(mouth_loss))
            # print('nose_loss[i]: ' + str(nose_loss))
            # print('leye_loss[i]: ' + str(leye_loss))
            # print('reye_loss[i]: ' + str(reye_loss))
            # print('============')

        loss_array = np.array(loss_array)
        tensor_distance_loss = K.variable(loss_array)

        # tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, loss_array])
        tensor_total_loss = tf.add(tensor_mean_square_error, tensor_distance_loss)
        return tensor_total_loss

    def __inceptionLoss_1(self, yTrue, yPred):
        return self.__soft_MSE(yTrue, yPred, 20)

    def __inceptionLoss_2(self, yTrue, yPred):
        return self.__soft_MSE(yTrue, yPred, 10)

    def __inceptionLoss_3(self, yTrue, yPred):
        return self.__soft_MSE(yTrue, yPred, 5)

    def __soft_MSE(self, yTrue, yPred, boundary_count, radius=0.01):
        yTrue_vector_batch = K.eval(yTrue)
        yPred_vector_batch = K.eval(yPred)

        out_batch_vector = []  # 50 *136
        for i in range(LearningConfig.batch_size):
            out_vector = []  # 136
            for j in range(LearningConfig.batch_size):
                if abs(yTrue_vector_batch[i, j] - yPred_vector_batch[i, j]) <= boundary_count * radius:
                    out_vector.append(0)
                else:
                    out_vector.append(1)
            out_batch_vector.append(out_vector)

        out_batch_vector = np.array(out_batch_vector)
        out_batch_tensor = K.variable(out_batch_vector)

        tensor_mean_square_error = K.mean(K.square(yPred - yTrue), axis=-1)
        tmp_mul = tf.multiply(tensor_mean_square_error, out_batch_tensor)
        return tmp_mul

    def __ASM(self, input_tensor, pca_postfix):
        print(pca_postfix)
        pca_utility = PCAUtility()
        image_utility = ImageUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug, pca_postfix=pca_postfix)

        input_vector_batch = K.eval(input_tensor)
        out_asm_vector = []
        for i in range(LearningConfig.batch_size):
            b_vector_p = self.calculate_b_vector(input_vector_batch[i], True, eigenvalues, eigenvectors, meanvector)
            # asm_vector = meanvector + np.dot(eigenvectors, b_vector_p)
            #
            # labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = \
            #     image_utility.create_landmarks_from_normalized(asm_vector, 224, 224, 112, 112)
            # imgpr.print_image_arr(i + 1, np.zeros(shape=[224,224,3]), landmark_arr_x_p, landmark_arr_y_p)

            out_asm_vector.append(meanvector + np.dot(eigenvectors, b_vector_p))

        out_asm_vector = np.array(out_asm_vector)

        tensor_out = K.variable(out_asm_vector)
        return tensor_out

    def __customLoss(self, yTrue, yPred):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        # yTrue = tf.constant([[1.0, 2.0, 3.0], [5.0, 4.0, 7.0]])
        # yPred = tf.constant([[2.0, 5.0, 6.0], [7.0, 3.0, 8.0]])
        # session = K.get_session()
        bias = 1
        tensor_mean_square_error = K.log((K.mean(K.square(yPred - yTrue), axis=-1) + bias))
        mse = K.eval(tensor_mean_square_error)
        # print("mse:")
        # print(mse)
        # print("---->>>")

        yPred_arr = K.eval(yPred)
        yTrue_arr = K.eval(yTrue)

        loss_array = []

        for i in range(LearningConfig.batch_size):
            asm_loss = 0

            truth_vector = yTrue_arr[i]
            predicted_vector = yPred_arr[i]

            b_vector_p = self.calculate_b_vector(predicted_vector, True, eigenvalues, eigenvectors, meanvector)
            y_pre_asm = meanvector + np.dot(eigenvectors, b_vector_p)

            for j in range(len(y_pre_asm)):
                asm_loss += (truth_vector[j] - y_pre_asm[j]) ** 2
            asm_loss /= len(y_pre_asm)

            asm_loss += bias + 1
            asm_loss = math.log(asm_loss, 10)
            asm_loss *= LearningConfig.regularization_term
            loss_array.append(asm_loss)
            print('mse[i]' + str(mse[i]))
            print('asm_loss[i]' + str(asm_loss))
            print('============')

        loss_array = np.array(loss_array)

        tensor_asm_loss = K.variable(loss_array)
        tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, tensor_asm_loss], axis=0)

        return tensor_total_loss

    def __customLoss_base(self, yTrue, yPred):
        pca_utility = PCAUtility()
        image_utility = ImageUtility()
        tf_record_utility = TFRecordUtility()

        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        # yTrue = tf.constant([[1.0, 2.0, 3.0], [5.0, 4.0, 7.0]])
        # yPred = tf.constant([[9.0, 1.0, 2.0], [7.0, 3.0, 8.0]])
        # session = K.get_session()

        tensor_mean_square_error = K.mean(K.square(yPred - yTrue), axis=-1)
        # tensor_mean_square_error = keras.losses.mean_squared_error(yPred, yTrue)
        mse = K.eval(tensor_mean_square_error)

        yPred_arr = K.eval(yPred)
        yTrue_arr = K.eval(yTrue)

        loss_array = []

        for i in range(LearningConfig.batch_size):
            asm_loss = 0

            truth_vector = yTrue_arr[i]
            predicted_vector = yPred_arr[i]

            b_vector_p = self.calculate_b_vector(predicted_vector, True, eigenvalues, eigenvectors, meanvector)
            y_pre_asm = meanvector + np.dot(eigenvectors, b_vector_p)

            """in order to test the results after PCA, you can use these lines of code"""
            # landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks_from_normalized(truth_vector, 224, 224, 112, 112)
            # image_utility.print_image_arr(i, np.ones([224, 224]), landmark_arr_x, landmark_arr_y)
            #
            # landmark_arr_xy_new, landmark_arr_x_new, landmark_arr_y_new= image_utility.create_landmarks_from_normalized(y_pre_asm, 224, 224, 112, 112)
            # image_utility.print_image_arr(i*100, np.ones([224, 224]), landmark_arr_x_new, landmark_arr_y_new)

            for j in range(len(y_pre_asm)):
                asm_loss += (truth_vector[j] - y_pre_asm[j]) ** 2
            asm_loss /= len(y_pre_asm)

            # asm_loss *= mse[i]
            # asm_loss *= LearningConfig.regularization_term

            loss_array.append(asm_loss)

            print('mse[i]' + str(mse[i]))
            print('asm_loss[i]' + str(asm_loss))
            print('============')

        loss_array = np.array(loss_array)
        tensor_asm_loss = K.variable(loss_array)

        # sum_loss_tensor = tf.add(tensor_mean_square_error, tensor_asm_loss)
        tensor_total_loss = tf.reduce_mean([tensor_mean_square_error, tensor_asm_loss], axis=0)

        # sum_loss = np.array(K.eval(tensor_asm_loss))
        # print(mse)
        # print(K.eval(tensor_mean_square_error))
        # print(K.eval(tensor_asm_loss))
        # print('asm_loss  ' + str(loss_array[0]))
        # print('mse_loss  ' + str(mse[0]))
        # print('sum_loss  ' + str(sum_loss[0]))
        # print('total_loss  ' + str(total_loss[0]))
        # print('      ')
        return tensor_total_loss

    def custom_teacher_student_loss_cos(self, lnd_img_map, img_path, teacher_models, teachers_weight_loss, bath_size,
                                        num_points, cos_weight):
        def loss(y_true, y_pred):
            cosine_loss = tf.keras.losses.cosine_similarity(axis=1)
            image_utility = ImageUtility()

            t0_model = teacher_models[0]
            l0_weight = teachers_weight_loss[0]

            t1_model = teacher_models[1]
            l1_weight = teachers_weight_loss[1]

            y_true_n = tf.reshape(y_true, [bath_size, num_points], name=None)
            imgs_address = self.get_y(y_true_n, lnd_img_map, img_path)
            imgs_batch = [np.array(Image.open(img_file)) / 255.0 for img_file in imgs_address]

            y_pred_T0 = np.array([t0_model.predict(np.expand_dims(img, axis=0))[0] for img in imgs_batch])
            y_pred_T1 = np.array([t1_model.predict(np.expand_dims(img, axis=0))[0] for img in imgs_batch])

            '''test teacher Nets'''
            # counter = 0
            # for pre_points in y_pred_T1:
            #     labels_predict_transformed, landmark_arr_x_p, landmark_arr_y_p = \
            #         image_utility.create_landmarks_from_normalized(pre_points, 224, 224, 112, 112)
            #     imgpr.print_image_arr((counter + 1) * 1000, imgs_batch[counter], landmark_arr_x_p, landmark_arr_y_p)
            #     counter += 1

            y_pred_T0_ten = K.variable(y_pred_T0)
            y_pred_T1_ten = K.variable(y_pred_T1)

            mse_te0 = K.mean(K.square(y_pred_T0_ten - y_true))
            mse_te0_cos = cosine_loss(y_pred_T0_ten, y_true)

            mse_te1 = K.mean(K.square(y_pred_T1_ten - y_true))
            mse_te1_cos = cosine_loss(y_pred_T1_ten, y_true)

            mse_main = K.mean(K.square(y_pred - y_true))
            mse_main_cos = cosine_loss(y_pred, y_true)

            return (mse_main + cos_weight * mse_main_cos) \
                   + l0_weight * (mse_te0 + cos_weight * mse_te0_cos) \
                   + l1_weight * (mse_te1 + cos_weight * mse_te1_cos)

        return loss

    def init_tensors(self, test):
        batchsize = LearningConfig.batch_size
        if test:
            batchsize = 1

        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug, )

        # print("predicted_tensor " + str(predicted_tensor.shape))
        # print("meanvector " + str(meanvector.shape))
        # print("eigenvalues " + str(eigenvalues.shape))
        # print("eigenvectors " + str(eigenvectors.shape))
        # print("-")

        self._meanvector_arr = np.tile(meanvector, (batchsize, 1))
        # meanvector_arr = np.tile(meanvector[None, :, None], (LearningConfig.batch_size, 1, 1))
        # print("meanvector_arr" + str(meanvector_arr.shape))

        self._eigenvalues_arr = np.tile(eigenvalues, (batchsize, 1))
        # eigenvalues_arr = np.tile(eigenvalues[None, :, None], (LearningConfig.batch_size, 1, 1))
        # print("eigenvalues_arr" + str(eigenvalues_arr.shape))

        self._eigenvectors_arr = np.tile(eigenvectors[None, :, :], (batchsize, 1, 1))
        # print("eigenvectors_arr" + str(eigenvectors_arr.shape))

        self._meanvector_tensor = tf.convert_to_tensor(self._meanvector_arr, dtype=tf.float32)
        self._eigenvalues_tensor = tf.convert_to_tensor(self._eigenvalues_arr, dtype=tf.float32)
        self._eigenvectors_tensor = tf.convert_to_tensor(self._eigenvectors_arr, dtype=tf.float32)

        self._eigenvectors_T = tf.transpose(self._eigenvectors_tensor, perm=[0, 2, 1])
        print("")

    def custom_activation(self, predicted_tensor):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        b_vector_tensor = self.calculate_b_vector_tensor(predicted_tensor, True, eigenvalues,
                                                         self._eigenvectors_tensor, self._meanvector_tensor)

        out = tf.add(tf.expand_dims(self._meanvector_tensor, 2), tf.matmul(self._eigenvectors_tensor, b_vector_tensor))
        out = tf.reshape(out, [LearningConfig.batch_size, 136])

        return out

    def calculate_b_vector_tensor(self, predicted_tensor, correction, eigenvalues, eigenvectors, mean_tensor):
        tmp1 = tf.expand_dims(tf.subtract(predicted_tensor, mean_tensor), 2)

        b_vector_tensor = tf.matmul(self._eigenvectors_T, tmp1)  # (50, 50, 1)

        return b_vector_tensor

        b_vector = np.squeeze(K.eval(b_vector_tensor), axis=2)[0]
        print("b_vector -> " + str(b_vector.shape))  # (50,)

        mul_arr = np.ones(b_vector.shape)
        add_arr = np.zeros(b_vector.shape)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    if b_item > lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    if b_item < -1 * lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = max(b_item, -1 * lambda_i_sqr)

                b_vector[i] = b_item
                i += 1

        mul_arr = np.tile(mul_arr, (LearningConfig.batch_size, 1))
        add_arr = np.tile(add_arr, (LearningConfig.batch_size, 1))

        # print(mul_arr)
        # print(add_arr)

        mul_tensor = tf.expand_dims(tf.convert_to_tensor(mul_arr, dtype=tf.float32), 2)
        add_arr = tf.expand_dims(tf.convert_to_tensor(add_arr, dtype=tf.float32), 2)

        # print("mul_tensor -> " + str(mul_tensor.shape))  # (50, 50, 1)
        # print("add_arr -> " + str(add_arr.shape))  # (50, 50, 1)

        tmp_mul = tf.multiply(b_vector_tensor, mul_tensor)
        tmp_add = tf.add(tmp_mul, add_arr)

        # print("add_arr -> " + str(add_arr.shape))  # (50, 50, 1)

        return tmp_add

    def custom_activation_test(self, predicted_tensor):
        pca_utility = PCAUtility()
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(DatasetName.ibug)

        b_vector_tensor = self.calculate_b_vector_tensor_test(predicted_tensor, True, eigenvalues,
                                                              self._eigenvectors_tensor, self._meanvector_tensor)

        out = tf.add(tf.expand_dims(self._meanvector_tensor, 2), tf.matmul(self._eigenvectors_tensor, b_vector_tensor))
        out = tf.reshape(out, [1, 136])

        return out

    def calculate_b_vector_tensor_test(self, predicted_tensor, correction, eigenvalues, eigenvectors, mean_tensor):

        # print("predicted_tensor -> " + str(predicted_tensor.shape))  # (50,)
        # print("eigenvalues -> " + str(eigenvalues.shape))  # (50,)
        # print("eigenvectors -> " + str(eigenvectors.shape))  # (50,)
        # print("mean_tensor -> " + str(mean_tensor.shape))  # (50,)

        tmp1 = tf.expand_dims(tf.subtract(predicted_tensor, mean_tensor), 2)
        # print("tmp1 -> " + str(tmp1.shape))  # (50,)

        b_vector_tensor = tf.matmul(self._eigenvectors_T, tmp1)  # (50, 50, 1)
        print("b_vector_tensor -> " + str(b_vector_tensor.shape))  # (50,)
        return b_vector_tensor

        # inputs = K.placeholder(shape=(None, 224, 224, 3))
        #
        # sess = K.get_session()
        # tmp22 = sess.run(inputs)
        # tmp22 = K.get_value(b_vector_tensor)

        tmp22 = K.eval(b_vector_tensor)

        # holder = tf.placeholder(tf.float32, shape=(None, 224,224,3))
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print("12")
        #     tmp22 = sess.run([b_vector_tensor], feed_dict=holder)
        # b_vector_tensor.eval(feed_dict=holder)

        print("tmp22 -> " + str(tmp22.shape))
        b_vector = np.squeeze(K.eval(b_vector_tensor), axis=2)
        print("b_vector -> " + str(b_vector.shape))  # (50,)

        mul_arr = np.ones(b_vector.shape)
        add_arr = np.zeros(b_vector.shape)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    if b_item > lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    if b_item < -1 * lambda_i_sqr:
                        mul_arr[i] = 0.0
                        add_arr[i] = lambda_i_sqr
                    b_item = max(b_item, -1 * lambda_i_sqr)

                b_vector[i] = b_item
                i += 1

        mul_arr = np.tile(mul_arr, (1, 1))
        add_arr = np.tile(add_arr, (1, 1))

        # print(mul_arr)
        # print(add_arr)

        mul_tensor = tf.expand_dims(tf.convert_to_tensor(mul_arr, dtype=tf.float32), 2)
        add_arr = tf.expand_dims(tf.convert_to_tensor(add_arr, dtype=tf.float32), 2)

        # print("mul_tensor -> " + str(mul_tensor.shape))  # (50, 50, 1)
        # print("add_arr -> " + str(add_arr.shape))  # (50, 50, 1)

        tmp_mul = tf.multiply(b_vector_tensor, mul_tensor)
        tmp_add = tf.add(tmp_mul, add_arr)

        print("tmp_add -> " + str(tmp_add.shape))  # (50, 50, 1)

        return tmp_add

    def calculate_b_vector(self, predicted_vector, correction, eigenvalues, eigenvectors, meanvector):
        tmp1 = predicted_vector - meanvector
        b_vector = np.dot(eigenvectors.T, tmp1)

        # put b in -3lambda =>
        if correction:
            i = 0
            for b_item in b_vector:
                lambda_i_sqr = 3 * math.sqrt(eigenvalues[i])

                if b_item > 0:
                    b_item = min(b_item, lambda_i_sqr)
                else:
                    b_item = max(b_item, -1 * lambda_i_sqr)
                b_vector[i] = b_item
                i += 1

        return b_vector

    def __reorder(self, input_arr):
        out_arr = []
        for i in range(68):
            out_arr.append(input_arr[i])
            k = 68 + i
            out_arr.append(input_arr[k])
        return np.array(out_arr)

    def test_pca_validity(self, pca_postfix):
        cnn_model = CNNModel()
        pca_utility = PCAUtility()
        tf_record_utility = TFRecordUtility()
        image_utility = ImageUtility()

        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca_obj(dataset_name=DatasetName.ibug,
                                                                         pca_postfix=pca_postfix)

        lbl_arr, img_arr, pose_arr = tf_record_utility.retrieve_tf_record(tfrecord_filename=IbugConf.tf_train_path,
                                                                          number_of_records=30, only_label=False)
        for i in range(20):
            b_vector_p = self.calculate_b_vector(lbl_arr[i], True, eigenvalues, eigenvectors, meanvector)
            lbl_new = meanvector + np.dot(eigenvectors, b_vector_p)

            labels_true_transformed, landmark_arr_x_t, landmark_arr_y_t = image_utility. \
                create_landmarks_from_normalized(lbl_arr[i], 224, 224, 112, 112)

            labels_true_transformed_pca, landmark_arr_x_pca, landmark_arr_y_pca = image_utility. \
                create_landmarks_from_normalized(lbl_new, 224, 224, 112, 112)

            image_utility.print_image_arr(i, img_arr[i], landmark_arr_x_t, landmark_arr_y_t)
            image_utility.print_image_arr(i * 1000, img_arr[i], landmark_arr_x_pca, landmark_arr_y_pca)

    _meanvector_arr = []
    _eigenvalues_arr = []
    _eigenvectors_arr = []
    _meanvector_tensor = None
    _eigenvalues_tensor = None
    _eigenvectors_tensor = None
    _eigenvectors_T = None
