from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize, CofwConf, WflwConf
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
from student_train import StudentTrainer
from test import Test
from train import Train

# from Train_Gan import TrainGan

if __name__ == '__main__':
    # tf_record_util = TFRecordUtility(136)
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

    # pca_utility.create_pca_from_points(DatasetName.cofw, 90)

    '''generate points with different accuracy'''
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug, pca_percentage=90)


    '''generate heatmap with different accuracy'''
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=85)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=90)
    # tf_record_util.generate_hm_and_save(dataset_name=DatasetName.ibug, pca_percentage=97)

    '''test heatmaps after creation'''
    # tf_record_util.load_hm_and_test(dataset_name=DatasetName.ibug)
    # pca_utility.test_pca_validity(DatasetName.cofw, 90)

    # mat = np.random.randint(0, 10, size=10)
    # cnn_model.generate_distance_matrix(mat)

    # cnn_model.init_for_test()

    # trg = TrainGan()
    # trg.create_seq_model()

    # test = Test(arch='efficientNet', num_output_layers=1, weight_fname='weights-181-0.00012.h5',
    #             dataset_name=DatasetName.ibug, point=True)

    # test = Test(arch='mobileNetV2', num_output_layers=1, weight_fname='weights-19--0.02211.h5',
    #             dataset_name=DatasetName.ibug)

    # test = Test(arch='mnv2_hm_r_v2', num_output_layers=1, weight_fname='weights-04-0.00995.h5', point=False)
    #

    trainer = Train(use_tf_record=True,
                    dataset_name=DatasetName.cofw,
                    custom_loss=False,
                    arch='efficientNet',
                    # arch='mnv2_hm_r_v2',
                    # arch='mb_mn',
                    inception_mode=False,
                    num_output_layers=1,
                    # weight='weights-60-0.00024.h5',
                    weight=None,
                    train_on_batch=False,
                    accuracy=100,
                    on_point=True)

    '''StudentTraining'''

    # st_trainer = StudentTrainer(dataset_name=DatasetName.wflw, arch="mobileNetV2")
    # st_trainer.train(teachers_arch=["efficientNet", "efficientNet"],
    #                  teachers_weight_files=["ds_wflw_ac_100_teacher.h5",
    #                                         "ds_wflw_ac_95_teacher.h5"],
    #                  teachers_weight_loss=[0.8, -1.5],
    #                  teachers_tf_train_paths=[WflwConf.tf_train_path, WflwConf.tf_train_path_95],
    #                  student_weight_file=None,
    #                  cos_weight=0.5
    #                  )





