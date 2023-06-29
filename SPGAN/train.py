import os
import ants
import torch
import argparse
import numpy as np
import torchvision
from tqdm import tqdm
from IPython import embed
from config.config import cfg
from models.SegNet import SegNet
from models.CycleGAN import CycleGAN
from dataset.dataset import InfantData
from utils.utils import weights_init, set_initial, calculate_patch_index


def main(args):
    # set initial checkpoint and testing results save path
    if cfg.general.resume_epoch == -1:
        set_initial(cfg)

    # Set numpy and torch seeds and device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(cfg.general.seed)

    # CycleGAN model initialize and load model parameters
    model = CycleGAN()
    model.initialize(args, cfg)

    if cfg.general.resume_epoch != -1:
        model.load_model()

    # Dataset initialize for training and validation
    training_set = InfantData(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.patch_size, fold=args.fold, type='train')
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=cfg.train.batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=True)

    validation_set = InfantData(cfg.general.root, cfg.general.file_list, crop_size=cfg.dataset.patch_size, fold=args.fold, type='val')
    infer_num = validation_set.__len__() // 4

    for epoch in range(0, cfg.train.num_epochs):
        loop = tqdm(training_loader, leave=True)
        print('Start train CycleGAN model at epoch: {}'.format(epoch))
        for idx, (img_06, seg_06, img_12, seg_12) in enumerate(loop):
            model.set_input(img_06, seg_06, img_12, seg_12)
            model.optimize_parameters()

            if idx == len(loop) // 2:
                real_A, real_B = model.real_A_img, model.real_B_img
                fake_A, fake_B = model.fake_A, model.fake_B
                rec_A, rec_B = model.rec_A, model.rec_B
                tmp = torch.cat([real_A, real_B, fake_A, fake_B, rec_A, rec_B], dim=0)
                grid = torchvision.utils.make_grid(tmp[:,:,:,:,48], normalize=True, nrow=cfg.train.batch_size)
                torchvision.utils.save_image(grid, os.path.join(cfg.general.save_dir, 'tmp', str(epoch)+'.jpg'))

        if epoch != 0 and epoch % cfg.train.save_epoch == 0:
            model.save_model(epoch)
            netG_A = model.netG_A
            netG_A.eval()
            netG_B = model.netG_B
            netG_B.eval()

            for idx in tqdm(range(infer_num)):
                save_path = os.path.join(cfg.general.save_dir, 'pred', 'chk_' + str(epoch))
                img_06, seg_06, img_12, seg_12, file_name, origin, spacing, direction = validation_set.__getitem__(idx)
                img_06, img_12, seg_12 = img_06.to(device), img_12.to(device), seg_12.to(device)
                img_06, img_12, seg_12 = img_06.unsqueeze(0), img_12.unsqueeze(0), seg_12.unsqueeze(0)
                img_06, img_12, seg_12 = img_06.unsqueeze(0), img_12.unsqueeze(0), seg_12.unsqueeze(0)

                B, C, W, H, D = img_06.shape

                pos = calculate_patch_index((W, H, D), cfg.dataset.patch_size)

                pred_A_rec = torch.zeros((1, W, H, D))
                pred_B_rec = torch.zeros((1, W, H, D))
                freq_rec = torch.zeros((1, W, H, D))

                for start_pos in pos:
                    if 'SPADE' not in args.GA2B:
                        img_06_patch = img_06[:, :, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                            start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1], start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]]
                        img_12_patch = img_12[:, :, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                            start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1], start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]]
                        pred_A_tmp = netG_A(img_12_patch)
                        pred_A_tmp = pred_A_tmp.cpu().detach()
                        pred_B_tmp = netG_B(img_06_patch)
                        pred_B_tmp = pred_B_tmp.cpu().detach()

                        pred_A_rec[:, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0], start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1], start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]] += pred_A_tmp[0, :, :, :, :]
                        pred_B_rec[:, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0], start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1], start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]] += pred_B_tmp[0, :, :, :, :]
                        freq_rec[:, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0], start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1], start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]] += 1

                    else:
                        img_06_patch = img_06[:, :, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                                       start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1],
                                       start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]]
                        img_12_patch = img_12[:, :, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                                       start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1],
                                       start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]]
                        seg_12_patch = seg_12[:, :, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                                       start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1],
                                       start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]]
                        pred_A_tmp = netG_A(img_12_patch, seg_12_patch)
                        pred_A_tmp = pred_A_tmp.cpu().detach()
                        pred_B_tmp = netG_B(img_06_patch)
                        pred_B_tmp = pred_B_tmp.cpu().detach()

                        pred_A_rec[:, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                        start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1],
                        start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]] += pred_A_tmp[0, :, :, :, :]
                        pred_B_rec[:, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                        start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1],
                        start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]] += pred_B_tmp[0, :, :, :, :]
                        freq_rec[:, start_pos[0]:start_pos[0] + cfg.dataset.patch_size[0],
                        start_pos[1]:start_pos[1] + cfg.dataset.patch_size[1],
                        start_pos[2]:start_pos[2] + cfg.dataset.patch_size[2]] += 1

                pred_A = pred_A_rec / freq_rec
                pred_A = pred_A.squeeze(0)
                pred_A = pred_A.numpy().astype(np.float32)
                pred_A = ants.from_numpy(pred_A, origin, spacing, direction)
                ants.image_write(pred_A, os.path.join(save_path, file_name + '_predASyn.nii.gz'))
                pred_B = pred_B_rec / freq_rec
                pred_B = pred_B.squeeze(0)
                pred_B = pred_B.numpy().astype(np.float32)
                pred_B = ants.from_numpy(pred_B, origin, spacing, direction)
                ants.image_write(pred_B, os.path.join(save_path, file_name + '_predBSyn.nii.gz'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infant Brain Synthesis using Implicit Prior Semantic Constraint')
    parser.add_argument('--platform', type=str, default='local', help='Specify compute platform')
    parser.add_argument('--fold', type=int, default=1, help='Specify training data folder')
    parser.add_argument('--save_path', type=str, default='CycleGAN', help='Specify training data folder')
    parser.add_argument('--file_list', type=str, default='file_list_06_12.csv', help='Specify training data folder')
    parser.add_argument('--GA2B', type=str, default='UNetGenerator',
                        help='Specify generator model for 12 month synthesis 06 month brain')
    parser.add_argument('--DA2B', type=str, default='PatchGANDiscriminator',
                        help='Specify discriminator model for six month synthesis 12 month brain')
    parser.add_argument('--GB2A', type=str, default='UNetGenerator',
                        help='Specify generator model for 12 month synthesis 06 month brain')
    parser.add_argument('--DB2A', type=str, default='PatchGANDiscriminator',
                        help='Specify discriminator model for 12 month synthesis 06 month brain')
    parser.add_argument('--SegLoss', type=bool, default=False,
                        help='Specify whether using segmentation model as additional loss for six month image synthesis 12 month')
    parser.add_argument('--features', type=int, default=32, help='# of filters of generator and discriminator')
    parser.add_argument('--lambda_idt', type=int, default=10, help='weight for identity loss')
    parser.add_argument('--lambda_A', type=int, default=10, help='weight for generator loss A')
    parser.add_argument('--lambda_B', type=int, default=10, help='weight for generator loss B')
    parser.add_argument('--lambda_co_A', type=int, default=10, help='weight for correlation loss A')
    parser.add_argument('--lambda_co_B', type=int, default=10, help='weight for correlation loss B')

    args = parser.parse_args()

    print('Start training on {} server'.format(args.platform))

    cfg.general.file_list = args.file_list
    cfg.general.save_dir = os.path.join(cfg.general.save_root, args.save_path)

    print('file list for training and validation: {}'.format(cfg.general.file_list))
    print('file path to save all images: {}'.format(cfg.general.save_dir))
    print('project level path: {}'.format(cfg.general.root))
    
    main(args)