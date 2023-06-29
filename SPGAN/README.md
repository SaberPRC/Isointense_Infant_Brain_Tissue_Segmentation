# Script for train the SPGAN
>This model is mainly refers to [CycleGAN](https://github.com/junyanz/CycleGAN) and [GauGAN](https://github.com/NVlabs/SPADE)
> 
Repo for train the bi-directional synthesize model (SPGAN) to transfer apperance between isointense infant brain and adult-like infant brain

***
<font color=gray size=3>Adult-like Phase and Multi-scale Assistance for Isointense Infant Brain Tissue Segmentation</font>

<font color=gray size=3>Jiameng Liu, Feihong Liu, Kaicong Sun, Mianxin Liu, Yuhang Sun, Yuyan Ge, and Dinggang Shen</font>

<font color=gray size=3>In MICCAI 2023</font> 
***

## [<font color=blue size=3>License</font> ](./LICENSE)

Copyright (C) ShanghaiTech University.

All rights reserved. Licensed under the GPL (General Public License)

The code is released for academic research use only. For commercial use or business inquiries, please contact JiamengLiu.PRC@gmail.com

***
1. Prepare data and organize your project following the below instruction
    ```shell
    root # root path for your project
    ├── csvfile # folder to save all data list
    │   └── file_list.csv # file list with 5-fold-split (IDs, time, fold)
    ├── data # folder to save all training, validation, and testing data
    │   ├── sub001
    │   │   ├── 06mo
    │   │   │   ├── brain.nii.gz
    │   │   │   └── tissue.nii.gz
    │   │   └── 12mo
    │   │       ├── brain.nii.gz
    │   │       └── tissue.nii.gz
    │   └── sub002
    └── results # folder to save results during training process
        ├── CycleGAN
        └── SPGAN
    ```

2. Training

* Script for training using SPGAN
   ```shell
  python train.py --save_path SPGAN --GA2B SPADEGenerator --GB2A UNetGenerator --resume -1 --batch_size 1 --SegLoss True  
  ```
   
* Script for training using CycleGAN
   ```shell
  python train.py --save_path SPGAN --GA2B UNetGenerator --GB2A UNetGenerator --resume -1 --batch_size 1 --SegLoss False  
  ```
  
***
