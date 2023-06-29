from easydict import EasyDict as edict

__C = edict()
cfg = __C


####### general parameters ######
__C.general = {}
__C.general.file_list = 'file_list.csv'
__C.general.root = '/path/to/project/root'
__C.general.save_root = '/path/to/save/root'

###### training parameters
__C.train = {}
__C.train.num_epochs = 3000
__C.train.batch_size=1

__C.train.lr = 2e-4
__C.train.save_epoch = 10

###### loss function setting ######
__C.loss = {}

__C.loss.name='HausdorffDistance'
__C.loss.weight = 0.4

# parameters for focal loss
__C.loss.obj_weight = ['99', '1']
__C.loss.gamma = 2

# resume_epoch == -1 training from scratch
__C.general.resume_epoch = -1


# random seed
__C.general.seed = 42


####### dataset parameters #######
__C.dataset = {}
# number of classes
__C.dataset.num_classes = 3
# number of modalities
__C.dataset.num_modalities = 2
# image resolution
__C.dataset.spacing = [1,1,1]
# cropped image patch size
__C.dataset.crop_size = [128, 128, 128]


# intensity normalize methods
# 1) FixedNormalize: use fixed mean and std to normalize image
# 2) AdaptiveNormalize: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.normalize = ['mean', 'std', 'is_clip']