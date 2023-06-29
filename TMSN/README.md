# Script for train the TMSN
 
>Repo for train the Transformer-based Multi-scale Assistance for Isointense Infant Brain Tissue Segmentation

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
  python trainSegNetMSAtt.py --platform bme --file_list file_list_06.csv --channel 2 --save_path SegNetMSAttReal2 --batch_size 4  
   ```
***
