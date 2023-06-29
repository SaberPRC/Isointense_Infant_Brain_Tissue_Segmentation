import os
import torch
import random
import itertools
from tqdm import tqdm
from IPython import embed
from models.SegNet import SegNet
from models.Discriminator import Discriminator, PatchGANDiscriminator
from models.Generator import ResnetGenerator, UNetGenerator, SPADEGenerator, SPADEGeneratorV1, SPADEGeneratorV2


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """Computes the Dice loss
        Notes: [Batch size,Num classes,Height,Width]
        Args:
            targets: a tensor of shape [B, H, W] or [B, 1, H, W].
            inputs: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model. (prediction)
            eps: added to the denominator for numerical stability.
        Returns:
            dice coefficient: the average 2 * class intersection over cardinality value
            for multi-class image segmentation
        """
        num_classes = inputs.size(1)

        true_1_hot = torch.eye(num_classes)[targets.long()]          # target을 one_hot vector로 만들어준다.

        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()   # [B,H,W,C] -> [B,C,H,W]
        # probas = F.softmax(inputs, dim=1)                     # preds를 softmax 취해주어 0~1사이 값으로 변환
        probas = inputs
        true_1_hot = true_1_hot.type(inputs.type())           # input과 type 맞춰주기
        dims = (0,) + tuple(range(2, targets.ndimension()))   # ?
        intersection = torch.sum(probas * true_1_hot, dims)   # TP
        cardinality = torch.sum(probas + true_1_hot, dims)    # TP + FP + FN + TN
        dice = ((2. * intersection + 1e-7) / (cardinality + 1e-7)).mean()

        return 1 - dice


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def define_G(args, device='cuda'):
    assert args.GA2B == 'UNetGenerator' or args.GA2B == 'SPADEGenerator', \
        'wrong Generator type for 12mo to 6mo'
    assert args.GB2A == 'ResNetGenerator' or args.GB2A == 'UNetGenerator', \
        'wrong Generator type for 6mo to 12mo'
    # Define six month 2  twelve month synthesis
    if args.GA2B == 'UNetGenerator':
        netG_A = UNetGenerator(in_channels=1, features=args.features)
        netG_A = torch.nn.DataParallel(netG_A)
        netG_A = netG_A.to(device)
    elif args.GA2B == 'SPADEGenerator':
        netG_A = SPADEGenerator(in_channels=1, features=args.features)
        netG_A = torch.nn.DataParallel(netG_A)
        netG_A = netG_A.to(device)

    # Define 06mo to 12mo brain synthesis
    if args.GB2A == 'UNetGenerator':
        netG_B = UNetGenerator(in_channels=1, features=args.features)
        netG_B = torch.nn.DataParallel(netG_B)
        netG_B = netG_B.to(device)
    elif args.GB2A == 'ResNetGenerator':
        netG_B = ResnetGenerator(input_nc=1, output_nc=1, ngf=args.features)
        netG_B = torch.nn.DataParallel(netG_B)
        netG_B = netG_B.to(device)

    return netG_A, netG_B


def define_D(args, device='cuda'):
    assert args.DA2B == 'Discriminator' or args.DA2B == 'PatchGANDiscriminator', \
        'wrong Discriminator type for 6mo to 12mo'
    assert args.DB2A == 'Discriminator' or args.DB2A == 'PatchGANDiscriminator', \
        'wrong Discriminator type for 12mo to 6mo'

    if args.DA2B == 'PatchGANDiscriminator':
        netD_A = PatchGANDiscriminator(input_nc=1)
        netD_A = torch.nn.DataParallel(netD_A)
        netD_A = netD_A.to(device)
        netD_B = PatchGANDiscriminator(input_nc=1)
        netD_B = torch.nn.DataParallel(netD_B)
        netD_B = netD_B.to(device)

    elif args.DA2B == 'Discriminator':
        netD_A = Discriminator(in_channels=1).to(device)
        netD_A = torch.nn.DataParallel(netD_A)
        netD_A = netD_A.to(device)
        netD_B = Discriminator(in_channels=1).to(device)
        netD_B = torch.nn.DataParallel(netD_B)
        netD_B = netD_B.to(device)

    return netD_A, netD_B


def SegLoss(args, cfg, device='cuda'):
    # Define segmentation model as additional loss to keep structure informations
    if args.SegLoss:
        SegModel = SegNet(in_channels=1, out_channels=4)
        SegModel = torch.nn.DataParallel(SegModel)
        SegModel = SegModel.to(device)
        SegModel.load_state_dict(torch.load(cfg.general.pretrain_seg))
        for p in SegModel.parameters():
            p.requires_grad = False
        SegModel.eval()
    return SegModel


class CycleGAN():
    def initialize(self, args, cfg):
        self.args = args
        self.cfg = cfg

        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        self.netG_A, self.netG_B = define_G(args, device='cuda')
        self.netD_A, self.netD_B = define_D(args, device='cuda')

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.criterionGAN = torch.nn.BCEWithLogitsLoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.L1_loss = torch.nn.L1Loss()
        self.dice_loss = DiceLoss()

        if args.SegLoss:
            self.criterionSeg = SegLoss(args, cfg)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=cfg.train.g_lr, betas=(cfg.train.g_betas[0], cfg.train.g_betas[1]))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=cfg.train.d_lr, betas=(cfg.train.g_betas[0], cfg.train.g_betas[1]))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

    def set_input(self, img_06, seg_06, img_12, seg_12, device='cuda'):
        self.real_A_img = img_12.to(device)
        self.real_A_seg = seg_12.to(device)
        self.real_B_img = img_06.to(device)
        self.real_B_seg = seg_06.to(device)

    def forward(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

        if 'SPADE' in self.args.GA2B:
            self.fake_B = self.netG_A(self.real_A_img, self.real_A_seg)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B_img)
            self.rec_B = self.netG_A(self.fake_A, self.real_A_seg)
        else:
            self.fake_B = self.netG_A(self.real_A_img)
            self.rec_A = self.netG_B(self.fake_B)
            self.fake_A = self.netG_B(self.real_B_img)
            self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_img, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_img, fake_A)

    def backward_G(self):
        lambda_idt = self.cfg.train.lambda_identity
        lambda_A = self.cfg.train.lambda_A
        lambda_B = self.cfg.train.lambda_B
        '''
        lambda_coA & lambda_coB
        '''
        lambda_co_A = self.cfg.train.lambda_co_A
        lambda_co_B = self.cfg.train.lambda_co_B

        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B_img, self.real_B_seg)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B_img) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A_img)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A_img) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(real_A_img))
        pred_fake_B = self.netD_A(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake_B, torch.ones_like(pred_fake_B))

        # GAN loss D_B(G_B(real_B_img))
        pred_fake_A = self.netD_B(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # Cycle consistence loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A_img) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B_img) * lambda_B

        if self.args.SegLoss:
            fake_out1, fake_out2, fake_out3, fake_out4, fake_out = self.criterionSeg(self.rec_A)
            real_out1, real_out2, real_out3, real_out4, real_out = self.criterionSeg(self.real_A_img)
            feature_loss = self.L1_loss(fake_out1, real_out1) + self.L1_loss(fake_out2, real_out2) + \
                           self.L1_loss(fake_out3, real_out3) + self.L1_loss(fake_out4, real_out4)
            prob_loss = self.L1_loss(fake_out, real_out)
            dice_loss = self.dice_loss(fake_out, self.real_A_seg.squeeze(1))
            self.loss_seg = (feature_loss + prob_loss + dice_loss) * self.cfg.train.lambda_seg
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_seg
        else:
            # combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward()

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # make models eval mode during test time
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def load_model(self):
        netG_A_path = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(self.cfg.resume_epoch) + '_gen_A2B.pth')
        netG_A_dict = torch.load(netG_A_path)
        self.netG_A.load_state_dict(netG_A_dict)

        netG_B_path = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(self.cfg.resume_epoch) + '_gen_B2A.pth')
        netG_B_dict = torch.load(netG_B_path)
        self.netG_B.load_state_dict(netG_B_dict)

        netD_A_path = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(self.cfg.resume_epoch) + '_disc_A2B.pth')
        netD_A_dict = torch.load(netD_A_path)
        self.netD_A.loadt_state_dict(netD_A_dict)

        netD_B_path = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(self.cfg.resume_epoch) + '_disc_B2A.pth')
        netD_B_dict = torch.load(netD_B_path)
        self.netD_B.load_state_dict(netD_B_dict)

    def save_model(self, epoch):
        save_path = os.path.join(self.cfg.general.save_dir, 'pred', 'chk_' + str(epoch))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path_gen = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(epoch) + '_gen_A2B.pth')
        torch.save(self.netG_A.state_dict(), save_path_gen)
        save_path_disc = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(epoch) + '_disc_A2B.pth')
        torch.save(self.netD_A.state_dict(), save_path_disc)

        save_path_gen = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(epoch) + '_gen_B2A.pth')
        torch.save(self.netG_B.state_dict(), save_path_gen)
        save_path_disc = os.path.join(self.cfg.general.save_dir, 'checkpoints', 'chk_' + str(epoch) + '_disc_B2A.pth')
        torch.save(self.netD_B.state_dict(), save_path_disc)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def __name__(self):
        return 'CycleGAN model'
























