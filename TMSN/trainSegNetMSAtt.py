'''
Repo. for train the synthesis based isointense phase infant brain tissue segmentation network
Using the Modified V-Net and supervised by focal & dice loss
Copy right: Jiameng Liu, ShanghaiTech University
Contact: JiamengLiu.PRC@gmail.com
'''

import os
import sys

import ants
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from IPython import embed
from config.config import cfg
from network.SegNetMSAtt import SegNetMultiScale
from torch.utils.data import DataLoader
from dataset.dataset import InfantSegBaseMS
from utils.loss import DiceLoss, FocalLoss
from torch.utils.tensorboard import SummaryWriter
from utils.utils import set_initial, calculate_patch_index, weights_init


def get_pred(img, model, batch_size):

    if len(img.shape) == 4:
        img = torch.unsqueeze(img, dim=0)

    B, C, W, H, D = img.shape

    m = nn.ConstantPad3d(16, 0)

    pos = calculate_patch_index((W, H, D), batch_size, overlap_ratio=0.4)
    pred_rec_s = torch.zeros((cfg.dataset.num_classes+1, W, H, D))
    pred_rec_b = torch.zeros((cfg.dataset.num_classes+1, W, H, D))

    freq_rec = torch.zeros((cfg.dataset.num_classes+1, W, H, D))

    for start_pos in pos:
        patch = img[:,:,start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]]
        model_out_s, model_out_b = model(patch)

        model_out_s = m(model_out_s)
        model_out_b = F.interpolate(model_out_b, size=batch_size)

        model_out_s = model_out_s.cpu().detach()
        model_out_b = model_out_b.cpu().detach()

        pred_rec_s[:, start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]] += model_out_s[0,:,:,:,:]
        pred_rec_b[:, start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]] += model_out_b[0,:,:,:,:]
        freq_rec[:, start_pos[0]:start_pos[0]+batch_size[0], start_pos[1]:start_pos[1]+batch_size[1], start_pos[2]:start_pos[2]+batch_size[2]] += 1

    pred_rec_s = pred_rec_s / freq_rec
    pred_rec_b = pred_rec_b / freq_rec

    pred_rec_s = pred_rec_s[:, 16:W-16, 16:H-16, 16:D-16]
    pred_rec_b = pred_rec_b[:, 16:W-16, 16:H-16, 16:D-16]

    return pred_rec_s, pred_rec_b


def _multi_layer_dice_coefficient(source, target, ep=1e-8):
    '''
    TODO: functions to calculate dice coefficient of multi class
    :param source: numpy array (Prediction)
    :param target: numpy array (Ground-Truth)
    :return: vector of dice coefficient
    '''
    class_num = target.max()+1

    source = source.astype(int)
    source = np.eye(class_num)[source]
    source = source[:,:,:,1:]
    source = source.reshape((-1, class_num-1))

    target = target.astype(int)
    target = np.eye(class_num)[target]
    target = target[:,:,:,1:]
    target = target.reshape(-1, class_num-1)

    intersection = 2 * np.sum(source * target, axis=0) + ep
    union = np.sum(source, axis=0) + np.sum(target, axis=0) + ep

    return intersection / union


def test(args, model, infer_data, infer_num, epoch, device = torch.device('cuda')):
    # initial model and set parameters
    model.eval()

    # setting save_path
    save_path = os.path.join(cfg.general.save_dir, 'pred', 'chk_'+str(epoch))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    rec = list()


    for idx in tqdm(range(infer_num//4)):
        # Get testing data info
        img_06, img_12, seg, img_name, origin, spacing, direction = infer_data.__getitem__(idx)

        img_06, img_12 = img_06.to(device), img_12.to(device)
        img_06, img_12 = img_06.unsqueeze(0), img_12.unsqueeze(0)
        if args.channel == 2:
            img = torch.cat([img_06, img_12], dim=0)
        else:
            img = img_06

        pred_s, pred_b = get_pred(img, model, cfg.dataset.crop_size)
        pred_s = pred_s.argmax(0)
        pred_s = pred_s.numpy().astype(np.float32)

        pred_b = pred_b.argmax(0)
        pred_b = pred_b.numpy().astype(np.float32)

        seg = seg.numpy().astype(int)

        dice_csf_s, dice_gm_s, dice_wm_s = _multi_layer_dice_coefficient(pred_s, seg)
        dice_csf, dice_gm, dice_wm = _multi_layer_dice_coefficient(pred_b, seg)

        ants_img_pred_seg_s = ants.from_numpy(pred_s, origin, spacing, direction)
        ants_img_pred_seg_b = ants.from_numpy(pred_b, origin, spacing, direction)

        rec.append([img_name, dice_csf_s, dice_gm_s, dice_wm_s, dice_csf, dice_gm, dice_wm, (dice_csf_s + dice_gm_s + dice_wm_s)/3])

        ants.image_write(ants_img_pred_seg_s, os.path.join(save_path, img_name + '_seg_s.nii.gz'))
        ants.image_write(ants_img_pred_seg_b, os.path.join(save_path, img_name + '_seg_b.nii.gz'))

    df = pd.DataFrame(rec, columns=['file_name', 'dice_csf_s', 'dice_gm_s', 'dice_wm_s', 'dice_csf_b', 'dice_gm_b',
                                    'dice_wm_b', 'dice_mean'])
    df.to_csv(os.path.join(save_path, str(epoch) + '.csv'), index=False)
    return None


def train(cfg, args):
    # set initial checkpoint and testing results save path
    if cfg.general.resume_epoch == -1:
        set_initial(cfg)

    # set initial tensorboard save path
    if not os.path.exists(os.path.join(cfg.general.save_dir, 'tensorboard')):
        os.mkdir(os.path.join(cfg.general.save_dir, 'tensorboard'))
    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))

    # Default tensor type
    torch.set_default_dtype(torch.float32)

    # Set computing device cpu or gpu
    device = torch.device('cuda')

    # Set numpy and torch seeds
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(cfg.general.seed)

    training_set = InfantSegBaseMS(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.crop_size, fold=args.fold, type='train')
    training_loader = DataLoader(training_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    iter_num = np.ceil(training_set.__len__() / cfg.train.batch_size)

    infer_data = InfantSegBaseMS(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.crop_size, fold=args.fold, type='val')
    infer_num = infer_data.__len__()

    # Init resume_spoch == -1, train from scratch
    if cfg.general.resume_epoch == -1:
        model = SegNetMultiScale(cfg.dataset.num_modalities, cfg.dataset.num_classes + 1)
        weights_init(model)
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = SegNetMultiScale(cfg.dataset.num_modalities, cfg.dataset.num_classes + 1)
        model_path = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_' + str(cfg.general.resume_epoch) + '.pth.gz')
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))

    # Optimization strategy
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, amsgrad=True)

    # Loss function for optimize
    loss_focal = FocalLoss(cfg.dataset.num_classes + 1)
    loss_dice = DiceLoss()

    for epoch in range(cfg.general.resume_epoch + 1, cfg.train.num_epochs):
        # training process
        model.train()
        for idx, (img_06, img_12, y) in enumerate(training_loader):
            tic = time.time()

            # start train one iteration
            if args.channel == 2:
                x = torch.cat([img_06, img_12], dim=1)
            else:
                x = img_06
            x, y = x.to(device), y.to(device)

            B, C, W, H, D = y.shape
            y_s = y[:, :, 16:W - 16, 16:H - 16, 16:D - 16]
            y_b = F.interpolate(y, size=[W - 32, H - 32, D - 32])

            y_s = y_s.squeeze(1)
            y_b = y_b.squeeze(1)

            optimizer.zero_grad()

            pred_s, pred_b = model(x)

            loss_seg_dice_s = loss_dice(pred_s, y_s)
            loss_seg_focal_s = loss_focal(pred_s, y_s)

            loss_seg_dice_b = loss_dice(pred_b, y_b)
            loss_seg_focal_b = loss_focal(pred_b, y_b)

            loss = 0.7 * (loss_seg_dice_s + loss_seg_focal_s * 10) + 0.3 * (loss_seg_dice_b + loss_seg_focal_b * 10)

            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss', loss.item(), epoch * iter_num + idx)
            writer.add_scalar('Loss/SegDice_s', loss_seg_dice_s.item(), epoch * iter_num + idx)
            writer.add_scalar('Loss/SegFocal_s', loss_seg_focal_s.item(), epoch * iter_num + idx)

            writer.add_scalar('Loss/SegDice_b', loss_seg_dice_b.item(), epoch * iter_num + idx)
            writer.add_scalar('Loss/SegFocal_b', loss_seg_focal_b.item(), epoch * iter_num + idx)

            toc = time.time()

            msg = 'epoch: {}, batch: {}, learning_rate: {}, loss: {:.4f}, loss_seg_dice: {:.4f}, loss_seg_focal: {:.4f}, loss_seg_dice_2: {:.4f}, loss_seg_focal_2: {:.4f}, time: {:.4f} s/iter' \
                .format(epoch, idx, optimizer.param_groups[0]['lr'], loss.item(), loss_seg_dice_s.item(), loss_seg_focal_s.item(), loss_seg_dice_b.item(), loss_seg_focal_b.item(), toc - tic)
            print(msg)

            if epoch != 0 and epoch % cfg.train.save_epoch == 0:
                save_path = os.path.join(cfg.general.save_dir, 'checkpoints', 'chk_' + str(epoch) + '.pth.gz')
                torch.save(model.state_dict(), save_path)
                if idx == 0 and epoch >= 200:
                    test(args, model, infer_data, infer_num, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infant brain tissue segmentation Experiment settings')
    parser.add_argument('--platform', type=str, default='local', help='specify compute platform')
    parser.add_argument('--fold', type=bool, default=1, help='specify testing fold')
    parser.add_argument('--save_path', type=str, default='SegNetMSAtt', help='specify results save folder')
    parser.add_argument('--file_list', type=str, default='file_list_sp.csv', help='specify training file list')
    parser.add_argument('--channel', type=int, default=1, help='number of input channels')
    parser.add_argument('--resume', type=int, default=-1, help='number of input channels')
    parser.add_argument('--batch_size', type=int, default=1, help='number of input channels')

    args = parser.parse_args()

    print('Start training on {} server'.format(args.platform))

    cfg.general.file_list = args.file_list
    cfg.general.save_dir = os.path.join(cfg.general.save_root, args.save_path)

    cfg.dataset.crop_size = [160, 160, 160]
    cfg.dataset.num_modalities = args.channel
    cfg.general.resume_epoch = args.resume
    cfg.train.batch_size=args.batch_size

    print('file list for training and validation: {}'.format(cfg.general.file_list))
    print('file path to save all images: {}'.format(cfg.general.save_dir))
    print('project level path: {}'.format(cfg.general.root))

    train(cfg, args)