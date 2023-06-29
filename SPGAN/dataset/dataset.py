import os
import torch
import random
import numpy as np
import pandas as pd

from dataset.basic import _ants_img_info, _normalize_z_score, _train_test_split
from dataset.basic import _random_seed, _mask_seed, _crop_and_convert_to_tensor


class InfantData(torch.utils.data.Dataset):
    def __init__(self, root, file_list, fold=1, type='train', crop_size=[128, 128, 96]):
        super().__init__()
        self.root = root
        self.type = type
        self.crop_size = crop_size
        
        file_list = pd.read_csv(os.path.join(self.root, 'csvfile', file_list))

        if self.type == 'train':
            file_list = file_list.loc[file_list['fold']!=fold]
            file_list = file_list.values.tolist()
            print('Training samples')
            print(len(file_list))
        elif self.type == 'val' or self.type == 'test':
            file_list = file_list.loc[file_list['fold'] == fold]
            file_list = file_list.values.tolist()

        self.file_list = file_list

    def __getitem__(self, idx):
        file_name = self.file_list[idx][0]
        img_06_path = os.path.join(self.root, 'data', file_name, '6mo', 'brain.nii.gz')
        seg_06_path = os.path.join(self.root, 'data', file_name, '6mo', 'tissue.nii.gz')
        _, _, _, img_06 = _ants_img_info(img_06_path)
        img_06 = _normalize_z_score(img_06)
        _, _, _, seg_06 = _ants_img_info(seg_06_path)

        img_12_path = os.path.join(self.root, 'data', file_name, '12_align_2_06', 'brain.nii.gz')
        seg_12_path = os.path.join(self.root, 'data', file_name, '12_align_2_06', 'tissue.nii.gz')
        _, _, _, img_12 = _ants_img_info(img_12_path)
        img_12 = _normalize_z_score(img_12)
        origin, spacing, direction, seg_12 = _ants_img_info(seg_12_path)

        if self.type == 'train':
            if random.random() > 0.2:
                start_pos = _mask_seed(seg_06, self.crop_size)
            else:
                start_pos = _random_seed(seg_06, self.crop_size)

            img_06_cropped = _crop_and_convert_to_tensor(img_06, start_pos, self.crop_size)
            seg_06_cropped = _crop_and_convert_to_tensor(seg_06, start_pos, self.crop_size)
            img_12_cropped = _crop_and_convert_to_tensor(img_12, start_pos, self.crop_size)
            seg_12_cropped = _crop_and_convert_to_tensor(seg_12, start_pos, self.crop_size)

            return img_06_cropped, seg_06_cropped, img_12_cropped, seg_12_cropped

        elif self.type == 'test' or self.type == 'val':
            img_06 = torch.from_numpy(img_06).type(torch.float32)
            seg_06 = torch.from_numpy(seg_06).type(torch.float32)
            img_12 = torch.from_numpy(img_12).type(torch.float32)
            seg_12 = torch.from_numpy(seg_12).type(torch.float32)
            return img_06, seg_06, img_12, seg_12, file_name, origin, spacing, direction

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    pass