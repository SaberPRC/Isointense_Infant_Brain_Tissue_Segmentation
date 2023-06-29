import os
import random

import numpy as np
import torch
from dataset.basic import _ants_img_info, _normalize_z_score, _train_test_split
from dataset.basic import _random_seed, _mask_seed, _crop_and_convert_to_tensor
from tqdm import tqdm


class InfantSegBaseMS(torch.utils.data.Dataset):
    def __init__(self, root, file_list, fold, type='train', crop_size=(128, 128, 128)):
        super().__init__()
        self.root = root
        self.type = type
        self.crop_size = crop_size
        file_list = os.path.join(root, 'csvfile', file_list)
        self.file_list = _train_test_split(file_list, fold, type)


    def __getitem__(self, idx):
        file_name, folder = self.file_list[idx][0], self.file_list[idx][1]
        data_key = file_name + folder
        img_06_path = os.path.join(self.root, 'data', file_name, folder, 'brain_06mo.nii.gz')
        origin, spacing, direction, img_06 = _ants_img_info(img_06_path)
        img_12_path = os.path.join(self.root, 'data', file_name, folder, 'brain_12mo.nii.gz')
        origin, spacing, direction, img_12 = _ants_img_info(img_12_path)
        seg_path = os.path.join(self.root, 'data', file_name, folder, 'tissue.nii.gz')
        origin, spacing, direction, seg = _ants_img_info(seg_path)


        img_06 = np.pad(img_06, ((16, 16), (16, 16), (16, 16)), 'constant')
        img_12 = np.pad(img_12, ((16, 16), (16, 16), (16, 16)), 'constant')

        if self.type == 'train':
            seg = np.pad(seg, ((16, 16), (16, 16), (16, 16)), 'constant')
            if random.random() > 0.2:
                start_pos = _mask_seed(seg, self.crop_size)
            else:
                start_pos = _random_seed(seg, self.crop_size)

            img_06_cropped = _crop_and_convert_to_tensor(img_06, start_pos, self.crop_size)
            img_12_cropped = _crop_and_convert_to_tensor(img_12, start_pos, self.crop_size)
            seg_cropped = _crop_and_convert_to_tensor(seg, start_pos, self.crop_size)

            return img_06_cropped, img_12_cropped, seg_cropped

        elif self.type == 'val' or self.type == 'test':
            img_06 = torch.from_numpy(img_06).type(torch.float32)
            img_12 = torch.from_numpy(img_12).type(torch.float32)
            seg = torch.from_numpy(seg).type(torch.float32)

            return img_06, img_12, seg, file_name, origin, spacing, direction

    def __len__(self):
        return len(self.file_list)


