import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class BasicBlock(nn.Module):
    # TODO: basic convolutional block, conv -> batchnorm -> activate
    def __init__(self, in_channels, out_channels, kernel_size, padding, activate=True, act='LeakyReLU'):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

        self.en_activate = activate

    def forward(self, x):
        if self.en_activate:
            return self.activate(self.bn(self.conv(x)))

        else:
            return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    # TODO: basic residual block established by BasicBlock
    def __init__(self, in_channels, out_channels, kernel_size, padding, nums, act='LeakyReLU'):
        '''
        TODO: initial parameters for basic residual network
        :param in_channels: input channel numbers
        :param out_channels: output channel numbers
        :param kernel_size: convoluition kernel size
        :param padding: padding size
        :param nums: number of basic convolutional layer
        '''
        super(ResidualBlock, self).__init__()

        layers = list()

        for _ in range(nums):
            if _ != nums - 1:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, True, act))

            else:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, False, act))

        self.do = nn.Sequential(*layers)

        if act == 'ReLU':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.activate = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.do(x)
        return self.activate(output + x)


class InputTransition(nn.Module):
    # TODO: input transition convert image to feature space
    def __init__(self, in_channels, out_channels):
        '''
        TODO: initial parameter for input transition <input size equals to output feature size>
        :param in_channels: input image channels
        :param out_channels: output feature channles
        '''
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        return out


class OutputTransition(nn.Module):
    # TODO: feature map convert to predict results
    def __init__(self, in_channels, out_channels, act='sigmoid'):
        '''
        TODO: initial for output transition
        :param in_channels: input feature channels
        :param out_channels: output results channels
        :param act: final activate layer sigmoid or softmax
        '''
        super(OutputTransition, self).__init__()
        assert act == 'sigmoid' or act =='softmax', \
            'final activate layer should be sigmoid or softmax, current activate is :{}'.format(act)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.act = act

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        if self.act == 'sigmoid':
            return self.sigmoid(out)
        elif self.act == 'softmax':
            return self.softmax(out)


class DownTransition(nn.Module):
    # TODO: fundamental down-sample layer <inchannel -> 2*inchannel>
    def __init__(self, in_channels, nums):
        '''
        TODO: intial for down-sample
        :param in_channels: inpuit channels
        :param nums: number of reisidual block
        '''
        super(DownTransition, self).__init__()

        out_channels = in_channels * 2
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)

    def forward(self, x):
        out = self.activate1(self.bn1(self.down(x)))
        out = self.residual(out)
        return out


class UpTransition(nn.Module):
    # TODO: fundamental up-sample layer (inchannels -> inchannels/2)
    def __init__(self, in_channels, out_channels, nums):
        '''
        TODO: initial for up-sample
        :param in_channels: input channels
        :param out_channels: output channels
        :param nums: number of residual block
        '''
        super(UpTransition, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels//2, kernel_size=2, stride=2, groups=1)
        self.bn = nn.BatchNorm3d(out_channels//2)
        self.activate = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)

    def forward(self, x, skip_x):
        out = self.activate(self.bn(self.conv1(x)))
        out = torch.cat((out,skip_x), 1)
        out = self.residual(out)

        return out


class transformer_layer(nn.Module):
    def __init__(self, nhead=8, batch_first=True, nhidden=512, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=512):
        super(transformer_layer, self).__init__()
        self.transformer = nn.Transformer(nhead=nhead, d_model=nhidden, dropout=True, batch_first=batch_first,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward)

    def forward(self, source, target):
        BS, CS, WS, HS, DS = source.shape
        BT, CT, WT, HT, DT = target.shape
        source = source.reshape(BS, CS, WS*HS*DS)
        target = target.reshape(BT, CT, WT*HT*DT)
        out = self.transformer(source, target)
        out = out.reshape(BT, CT, WT, HT, DT)
        return out


class SegNetMultiScale(nn.Module):
    # TODO: fundamental segmentation framework
    # Multi-Scale strategy using different crop size and normalize to same size
    def __init__(self, in_channels, out_channels, nhead=8):
        super(SegNetMultiScale, self).__init__()
        self.in_tr_s = InputTransition(in_channels, 16)
        self.in_tr_b = InputTransition(in_channels, 16)

        self.down_32_s = DownTransition(16, 1)
        self.down_32_b = DownTransition(16, 1)
        self.fusion_32 = BasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.down_64_s = DownTransition(32, 1)
        self.down_64_b = DownTransition(32, 1)
        self.fusion_64 = BasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.down_128_s = DownTransition(64, 2)
        self.down_128_b = DownTransition(64, 2)
        self.down_128_fuse = transformer_layer(nhead=nhead, nhidden=4096, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=512)

        self.down_256_s = DownTransition(128, 2)
        self.down_256_b = DownTransition(128, 2)
        self.down_256_fuse = transformer_layer(nhead=nhead, nhidden=512, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=512)

        self.up_256_s = UpTransition(256, 256, 2)
        self.up_256_b = UpTransition(256, 256, 2)
        self.up_256_fuse = transformer_layer(nhead=nhead, nhidden=4096, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=512)

        self.up_128_s = UpTransition(256, 128, 2)
        self.up_128_b = UpTransition(256, 128, 2)
        self.fusion_up_128 = BasicBlock(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.up_64_s = UpTransition(128, 64, 1)
        self.up_64_b = UpTransition(128, 64, 1)
        self.fusion_up_64 = BasicBlock(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.up_32_s = UpTransition(64, 32, 1)
        self.up_32_b = UpTransition(64, 32, 1)
        self.fusion_up_32 = BasicBlock(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.out_tr_s = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_b = OutputTransition(32, out_channels, 'softmax')

    def forward(self, x):
        B, C, W, H, D = x.shape
        B_s, C_s, W_s, H_s, D_s = B, C, W - 32, H - 32, D - 32

        x_s = x[:, :, 16:W - 16, 16:H - 16, 16:D - 16]
        x_b = F.interpolate(x, size=[W_s, H_s, D_s])

        out_16_s = self.in_tr_s(x_s)
        out_16_b = self.in_tr_b(x_b)

        out_32_s = self.down_32_s(out_16_s)
        out_32_b = self.down_32_b(out_16_b)

        out_32_s = torch.cat([out_32_s, F.interpolate(out_32_b[:, :, 7:57, 7:57, 7:57], size=[64, 64, 64])], dim=1)
        out_32_s = self.fusion_32(out_32_s)

        out_64_s = self.down_64_s(out_32_s)
        out_64_b = self.down_64_b(out_32_b)

        out_64_s = torch.cat([out_64_s, F.interpolate(out_64_b[:, :, 4:28, 4:28, 4:28], size=[32, 32, 32])], dim=1)
        out_64_s = self.fusion_64(out_64_s)

        out_128_s = self.down_128_s(out_64_s)
        out_128_b = self.down_128_b(out_64_b)
        out_128_s = self.down_128_fuse(F.interpolate(out_128_b[:, :, 2:14, 2:14, 2:14], size=[16, 16, 16]), out_128_s)

        out_256_s = self.down_256_s(out_128_s)
        out_256_b = self.down_256_b(out_128_b)
        out_256_s = self.down_256_fuse(F.interpolate(out_256_b[:, :, 1:7, 1:7, 1:7], size=[8, 8, 8]), out_256_s)

        out_s = self.up_256_s(out_256_s, out_128_s)
        out_b = self.up_256_b(out_256_b, out_128_b)
        out_s = self.up_256_fuse(F.interpolate(out_b[:, :, 2:14, 2:14, 2:14], size=[16, 16, 16]), out_s)

        out_s = self.up_128_s(out_s, out_64_s)
        out_b = self.up_128_b(out_b, out_64_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 4:28, 4:28, 4:28], size=[32, 32, 32])], dim=1)
        out_s = self.fusion_up_128(out_s)

        out_s = self.up_64_s(out_s, out_32_s)
        out_b = self.up_64_b(out_b, out_32_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 7:57, 7:57, 7:57], size=[64, 64, 64])], dim=1)
        out_s = self.fusion_up_64(out_s)

        out_s = self.up_32_s(out_s, out_16_s)
        out_b = self.up_32_b(out_b, out_16_b)

        out_s = torch.cat([out_s, F.interpolate(out_b[:, :, 13:115, 13:115, 13:115], size=[128, 128, 128])], dim=1)
        out_s = self.fusion_up_32(out_s)

        out_s = self.out_tr_s(out_s)
        out_b = self.out_tr_b(out_b)

        return out_s, out_b


if __name__ == '__main__':
    x = torch.randn(1, 1, 160, 160, 160)
    model = SegNetMultiScale(1, 4)
    x = x.to('cuda')
    model = model.to('cuda')
    out = model(x)
    embed()