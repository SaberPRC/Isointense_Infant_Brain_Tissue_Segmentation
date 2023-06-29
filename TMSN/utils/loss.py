import torch
import torch.nn as nn
import numpy as np
import math
from IPython import embed
from sklearn.utils.extmath import cartesian


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) / class_num
        else:
            assert len(alpha) == class_num
            self.alpha = torch.FloatTensor(alpha)
            self.alpha = self.alpha.unsqueeze(1)
            self.alpha = self.alpha/self.alpha.sum()

        self.alpha = self.alpha.cuda()
        # self.alpha = self.alpha
        self.gamma = gamma
        self.size_average = size_average
        self.class_num = class_num
        self.one_hot_codes = torch.eye(self.class_num).cuda()
        # self.one_hot_codes = torch.eye(self.class_num)

    def forward(self, input, target):
        # the input size should be one of the follows
        # 1. B*class_num
        # 2. B, class_num, x, y
        # 3. B, class_num, x, y, z
        assert input.dim() == 2 or input.dim() == 4 or input.dim() == 5
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(input.numel()//self.class_num, self.class_num)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input = input.view(input.numel()//self.class_num, self.class_num)


        # the size of target tensor should be
        # 1. B, 1 or B
        # 2. B, 1, x, y or B, x, y
        # 3. B, 1, x, y, z or B, x, y, z
        target = target.contiguous()
        target = target.long().view(-1)

        mask = self.one_hot_codes[target.data]
        # mask = Variable(mask, requires_grad=False)

        alpha = self.alpha[target.data]
        # alpha = Variable(alpha, requires_grad=False)

        probs = (input*mask).sum(1).view(-1, 1) + 1e-10
        log_probs = probs.log()

        if self.gamma > 0:
            batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_probs
        else:
            batch_loss = -alpha * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super(DiceLoss, self).__init__()
        self.size_average = size_average


    def forward(self, inputs, targets):
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
        true_1_hot = torch.eye(num_classes)[targets.long().cpu()]          # target을 one_hot vector로 만들어준다.

        true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()   # [B,H,W,C] -> [B,C,H,W]
        # probas = F.softmax(inputs, dim=1)                     # preds를 softmax 취해주어 0~1사이 값으로 변환
        probas = inputs
        true_1_hot = true_1_hot.type(inputs.type())           # input과 type 맞춰주기
        dims = (0,) + tuple(range(2, targets.ndimension()))   # ?
        intersection = torch.sum(probas * true_1_hot, dims)   # TP
        cardinality = torch.sum(probas + true_1_hot, dims)    # TP + FP + FN + TN
        dice = ((2. * intersection + 1e-7) / (cardinality + 1e-7)).mean()
        return 1- dice