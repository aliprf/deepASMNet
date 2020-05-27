
from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test
from Train_Gan import TrainGan

if __name__ == '__main__':
    tf_record_util = TFRecordUtility()
    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # tf_record_util.test_hm_accuracy()

    # tf_record_util.create_adv_att_img_hm()

    '''create and save PCA objects'''
    # pca_utility.create_pca_from_npy(DatasetName.ibug, 85)
    # pca_utility.create_pca_from_npy(DatasetName.ibug, 90)
    # pca_utility.create_pca_from_npy(DatasetName.ibug, 95)
    # pca_utility.create_pca_from_npy(DatasetName.ibug, 97)

    # pca_utility.create_pca_from_points(DatasetName.ibug, 85)

    '''generate points with different accuracy'''
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug, pca_percentage=85)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug, pca_percentage=90)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug, pca_percentage=97)

    '''generate heatmap with different accuracy'''
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=85)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=90)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=97)

    '''test heatmaps after creation'''
    # tf_record_util.load_hm_and_test(dataset_name=DatasetName.ibug)
    # pca_utility.test_pca_validity(DatasetName.ibug, 90)

    # mat = np.random.randint(0, 10, size=10)
    # cnn_model.generate_distance_matrix(mat)

    # cnn_model.init_for_test()

    # trg = TrainGan()
    # trg.create_seq_model()

    # test = Test(arch='efficientNet', num_output_layers=1, weight_fname='weights-04-0.01070.h5', point=True)
    # test = Test(arch='mnv2_hm_r_v2', num_output_layers=1, weight_fname='weights-04-0.00995.h5', point=False)
    #

    trainer = Train(use_tf_record=False,
                    dataset_name=DatasetName.ibug,
                    custom_loss=False,
                    arch='efficientNet',
                    # arch='mnv2_hm_r_v2',
                    # arch='mb_mn',
                    inception_mode=False,
                    num_output_layers=1,
                    # weight='weights-145-0.00011.h5',
                    weight=None,
                    train_on_batch=False,
                    accuracy=90,
                    on_point=True)






