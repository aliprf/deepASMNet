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
    # pca_utility.create_pca_from_points(DatasetName.ibug, 85)
    # pca_utility.create_pca_from_points(DatasetName.ibug, 90)
    # pca_utility.create_pca_from_points(DatasetName.ibug, 97)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug)

    # tf_record_util.load_hm_and_test(dataset_name=DatasetName.ibug)

    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=85)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=90)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=97)

    # pca_utility.test_pca_validity(DatasetName.ibug, 90)

    # tf_record_util.generate_hm_and_save()

    # tf_record_util.retrive_hm_and_test()

    # mat = np.random.randint(0, 10, size=10)
    # cnn_model.generate_distance_matrix(mat)

    # hm = np.random.randint(0, 10, size=(10, 10, 68))
    # tf_record_util.from_heatmap_to_point(hm, 5)
    #
    # cnn_model.test_result(multitask=True, singletask=False, pose=True)
    # cnn_model.test_result(multitask=False, singletask=True, pose=True)
    # cnn_model.test_result_distilation()

    # cnn_model.init_for_test()

    # cnn_model.train(dataset_name=DatasetName.ibug, asm_loss=True)

    # cnn_model.train_new(dataset_name=DatasetName.ibug, custom_loss=False, arch='mn_r', inception_mode=True) # mn, mn_r
    # cnn_model.test_new(arch='mn_r') # mn, mn_r

    # trg = TrainGan()
    # trg.create_seq_model()

    # test = Test(arch='mb_mn', num_output_layers=3, weight_fname='weights-101-0.00021_asm_model.h5')

    trainer = Train(use_tf_record=False,
                    dataset_name=DatasetName.ibug,
                    custom_loss=False,
                    arch='mn_asm_0',
                    # arch='mb_mn',
                    inception_mode=False,
                    num_output_layers=1,
                    weight=None,
                    train_on_batch=False,
                    accuracy=97)






