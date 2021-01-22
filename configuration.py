class DatasetName:
    affectnet = 'affectnet'
    w300 = 'w300'
    ibug = 'ibug'
    aflw = 'aflw'
    aflw2000 = 'aflw2000'
    cofw = 'cofw'
    wflw = 'wflw'



class DatasetType:
    data_type_train = 0
    data_type_validation = 1
    data_type_test = 2


class LearningConfig:
    CLR_METHOD = "triangular"
    MIN_LR = 1e-5
    MAX_LR = 1e-2
    STEP_SIZE = 10
    # batch_size = 2
    batch_size = 110
    # steps_per_validation_epochs = 5

    epochs = 1500
    # landmark_len = 136
    # point_len = 68
    pose_len = 3

    reg_term_ASM = 0.8



class InputDataSize:
    image_input_size = 224
    # landmark_len = 136
    landmark_face_len = 54
    landmark_nose_len = 18
    landmark_eys_len = 24
    landmark_mouth_len = 40
    pose_len = 3


class AffectnetConf:
    csv_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/training.csv'
    csv_evaluate_path = '/media/ali/extradata/facial_landmark_ds/affectNet/validation.csv'
    csv_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.csv'

    tf_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/train.tfrecords'
    tf_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/eveluate.tfrecords'
    tf_evaluation_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.tfrecords'

    sum_of_train_samples = 200000  # 414800
    sum_of_test_samples = 30000
    sum_of_validation_samples = 5500

    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/affectNet/Manually_Annotated_Images/'


class Multipie:
    lbl_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/MPie_Labels/labels/all/'
    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/'

    origin_number_of_all_sample = 2000
    origin_number_of_train_sample = 1950
    origin_number_of_evaluation_sample = 50
    augmentation_factor = 100


class W300Conf:
    tf_common = '/media/ali/data/test_common.tfrecords'
    tf_challenging = '/media/ali/data/test_challenging.tfrecords'
    tf_full = '/media/ali/data/test_full.tfrecords'

    img_path_prefix_common = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/common/'
    img_path_prefix_challenging = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/challenging/'
    img_path_prefix_full = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/full/'

    number_of_all_sample_common = 554
    number_of_all_sample_challenging = 135
    number_of_all_sample_full = 689


class WflwConf:
    Wflw_prefix_path = '/media/data3/ali/FL/new_data/wflw/'  # --> Zeus
    # Wflw_prefix_path = '/media/data2/alip/FL/new_data/wflw/'  # --> Atlas
    # Wflw_prefix_path = '/media/ali/data/wflw/'  # --> local

    img_path_prefix = Wflw_prefix_path + 'all/'
    rotated_img_path_prefix = Wflw_prefix_path + '0_rotated/'
    train_images_dir = Wflw_prefix_path + '1_train_images_pts_dir/'
    normalized_points_npy_dir = Wflw_prefix_path + '2_normalized_npy_dir/'
    pose_npy_dir = Wflw_prefix_path + '4_pose_npy_dir/'

    '''     augmented version'''
    augmented_train_pose = Wflw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Wflw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_atr = Wflw_prefix_path + 'training_set/augmented/atrs/'
    augmented_train_image = Wflw_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = Wflw_prefix_path + 'training_set/augmented/tf/'
    '''     original version'''
    no_aug_train_annotation = Wflw_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_atr = Wflw_prefix_path + 'training_set/no_aug/atrs/'
    no_aug_train_pose = Wflw_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = Wflw_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = Wflw_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 7500
    orig_number_of_test = 2500

    number_of_all_sample = 270956  # just images. dont count both img and lbls
    number_of_train_sample = number_of_all_sample * 0.95  # 95 % for train
    number_of_evaluation_sample = number_of_all_sample * 0.05  # 5% for evaluation

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 15  # create . image from 1
    num_of_landmarks = 98

class CofwConf:
    # Cofw_prefix_path = '/media/data3/ali/FL/new_data/cofw/'  # --> zeus
    Cofw_prefix_path = '/media/data2/alip/FL/new_data/cofw/'  # --> atlas
    # Cofw_prefix_path = '/media/ali/data/new_data/cofw/'  # --> local

    augmented_train_pose = Cofw_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = Cofw_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_image = Cofw_prefix_path + 'training_set/augmented/images/'

    augmented_train_tf_path = Cofw_prefix_path + 'training_set/augmented/tf/'
    no_aug_train_tf_path = Cofw_prefix_path + 'training_set/no_aug/tf/'

    tf_train_path_95 = Cofw_prefix_path + 'train_90.tfrecords'
    tf_evaluation_path_95 = Cofw_prefix_path + 'evaluation_90.tfrecords'

    orig_number_of_training = 1345
    orig_number_of_test = 507

    augmentation_factor = 10
    number_of_all_sample = orig_number_of_training * augmentation_factor  # afw, train_helen, train_lfpw
    number_of_train_sample = number_of_all_sample * 0.95  # 95 % for train
    number_of_evaluation_sample = number_of_all_sample * 0.05  # 5% for evaluation

    augmentation_factor = 5  # create . image from 1
    augmentation_factor_rotate = 30  # create . image from 1
    num_of_landmarks = 29


class IbugConf:
    w300w_prefix_path = '/media/data3/ali/FL/new_data/300W/'  # --> zeus
    # w300w_prefix_path = '/media/data2/alip/FL/new_data/300W/'  # --> atlas
    # w300w_prefix_path = '/media/ali/data/new_data/300W/'  # --> local

    orig_300W_train = w300w_prefix_path + 'orig_300W_train/'
    augmented_train_pose = w300w_prefix_path + 'training_set/augmented/pose/'
    augmented_train_annotation = w300w_prefix_path + 'training_set/augmented/annotations/'
    augmented_train_image = w300w_prefix_path + 'training_set/augmented/images/'
    augmented_train_tf_path = w300w_prefix_path + 'training_set/augmented/tf/'

    no_aug_train_annotation = w300w_prefix_path + 'training_set/no_aug/annotations/'
    no_aug_train_pose = w300w_prefix_path + 'training_set/no_aug/pose/'
    no_aug_train_image = w300w_prefix_path + 'training_set/no_aug/images/'
    no_aug_train_tf_path = w300w_prefix_path + 'training_set/no_aug/tf/'

    orig_number_of_training = 3148
    orig_number_of_test_full = 689
    orig_number_of_test_common = 554
    orig_number_of_test_challenging = 135

    '''after augmentation'''
    number_of_all_sample = 134688   # afw, train_helen, train_lfpw
    number_of_train_sample = number_of_all_sample * 0.95  # 95 % for train
    number_of_evaluation_sample = number_of_all_sample * 0.05  # 5% for evaluation

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 20  # create . image from 1
    num_of_landmarks = 68

