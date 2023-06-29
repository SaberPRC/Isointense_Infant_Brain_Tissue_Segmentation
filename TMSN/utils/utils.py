import os
import ants
import torch
import shutil
from itertools import product

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def set_initial(cfg):
    if cfg.general.resume_epoch < 0 and os.path.isdir(cfg.general.save_dir):
        print('Found non-empty save dir {}, \n {} to delete all files, {} to continue:'.format(cfg.general.save_dir,
                                                                                            'yes', 'no'), end=' ')
        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(cfg.general.save_dir)
        elif choice == 'no':
            pass
        else:
            raise ValueError('choice error')

    if not os.path.exists(cfg.general.save_dir):
        os.mkdir(cfg.general.save_dir)
        os.mkdir(os.path.join(cfg.general.save_dir, 'checkpoints'))
        os.mkdir(os.path.join(cfg.general.save_dir, 'pred'))

    if not os.path.isdir(os.path.join(cfg.general.save_dir, 'checkpoints')):
        os.mkdir(os.path.join(cfg.general.save_dir, 'checkpoints'))

    if not os.path.isdir(os.path.join(cfg.general.save_dir, 'pred')):
        os.mkdir(os.path.join(cfg.general.save_dir, 'pred'))


def calculate_patch_index(target_size, patch_size, overlap_ratio = 0.25):
    shape = target_size

    gap = int(patch_size[0] * (1-overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0]-patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1]-patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2]-patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos