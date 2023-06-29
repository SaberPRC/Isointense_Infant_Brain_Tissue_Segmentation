# Isointense Infant Brain Tissue Segmentation
Repo for isointense infant brain tissue segmentation

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

## Installation
***

Clone this repo
```shell
git clone https://github.com/SaberPRC/Isointense_Infant_Brain_Tissue_Segmentation.git
```
This code requires Pytorch 1.11.0 and Python 3.8.13, please install dependencies by
```shell
pip install -r requirements.txt
```
For reimplement the results as shown in this paper, you need prepare the data from National Database for Autism Research ([NDAR](https://healthdata.gov/dataset/National-Database-for-Autism-Research-NDAR-/7ue5-z77y/data))


***
## Training for SPAGN
> Please refer to [SPGAN/README](./SPGAN/README.md)

This model is mainly refers to [CycleGAN](https://github.com/junyanz/CycleGAN) and [GauGAN](https://github.com/NVlabs/SPADE)


## Training for TMSN
> Please refer to [TMSN/README](./TMSN/README.md)