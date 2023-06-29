from easydict import EasyDict as edict

__C = edict()
cfg = __C


####### general parameters ######
__C.general = {}
__C.general.file_list = 'file_list.csv'
__C.general.root = '/path/to/project/root'
__C.general.save_root = '/path/to/save/path/root'
__C.general.pretrain_seg = '/path/to/pretrained/adult-like/tissue/segmentation/model/path'

###### training parameters
__C.train = {}
__C.train.num_epochs = 2000
__C.train.batch_size= 2

__C.train.g_lr = 2e-4
__C.train.g_betas = [0.001, 0.999]

__C.train.d_lr = 2e-4
__C.train.d_betas = [0.001, 0.999]

__C.train.save_epoch = 10

__C.train.lambda_identity = 10
__C.train.lambda_A = 10
__C.train.lambda_co_A = 10
__C.train.lambda_B = 10
__C.train.lambda_co_B = 10
__C.train.lambda_seg = 0.1


# resume_epoch == -1 training from scratch
__C.general.resume_epoch = -1

# random seed
__C.general.seed = 42

####### dataset parameters #######
__C.dataset = {}
# number of classes
__C.dataset.num_classes = 3
# number of modalities
__C.dataset.num_modalities = 1
# image resolution
__C.dataset.spacing = [1,1,1]
# cropped image patch size
__C.dataset.patch_size = [128, 128, 96]

__C.model = {}
__C.model.nf = 16
__C.model.semantic_nc = 4
__C.model.L1_LAMBDA=100
