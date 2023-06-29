import torch
import functools
import numpy as np
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect"),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv3d(in_channels, out_channels, 3, 1, 1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2),
            )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.main(x) + x


class SPADE(nn.Module):
    def __init__(self, in_channels, semantic_label=1):
        super(SPADE, self).__init__()

        n_hidden = in_channels

        self.initial = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.mlp_shared = nn.Sequential(
            nn.Conv3d(semantic_label, n_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mlp_gamma = nn.Conv3d(n_hidden, in_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(n_hidden, in_channels, kernel_size=3, padding=1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, segmap):
        # step 1: Generate param free convolutional layer
        x = self.initial(x)
        segmap = torch.nn.functional.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = x * (1 + gamma*0.01) + beta * 0.01
        out = self.act(out)

        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type)

    def build_conv_block(self, dim, padding_type):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       nn.InstanceNorm3d(dim),
                       nn.LeakyReLU(0.2)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       nn.InstanceNorm3d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    # TODO: Resnet generator
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer='instance', n_blocks=6, padding_type='reflect'):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.norm = norm_layer

        # TODO: first define two 3 by 3 convolutional layer
        model = [nn.ReplicationPad3d(1),
                 nn.Conv3d(input_nc, ngf, kernel_size=3, padding=0,),
                 nn.InstanceNorm3d(ngf),
                 nn.LeakyReLU(0.2),

                 nn.ReplicationPad3d(1),
                 nn.Conv3d(ngf, ngf, kernel_size=3, padding=0, ),
                 nn.InstanceNorm3d(ngf),
                 nn.LeakyReLU(0.2),
                 ]

        n_downsampling = 3

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      nn.InstanceNorm3d(ngf * mult * 2),
                      nn.LeakyReLU(0.2)]
            model += [ResnetBlock(ngf*mult*2, padding_type=padding_type)]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2, mode='trilinear'),
                      nn.Conv3d(ngf*mult, int(ngf * mult / 2), kernel_size=3, padding=1),
                      nn.InstanceNorm3d(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.2)]
            model += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type)]

        model += [nn.Conv3d(ngf, output_nc, kernel_size=1, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    # TODO: UNet generator with skip connection
    def __init__(self, in_channels=1, features=16):
        super().__init__()
        self.in_tr = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm3d(features),
            nn.LeakyReLU(0.2),
        )
        self.initial_down = nn.Sequential(
            nn.Conv3d(features, features * 2, 4, 2, 1, padding_mode="reflect"),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 8, features * 16, down=True, act="leaky", use_dropout=False)

        self.bottleneck1 = bottleneck(features * 16, features * 16)
        self.bottleneck2 = bottleneck(features * 16, features * 16)
        self.bottleneck3 = bottleneck(features * 16, features * 16)

        self.up1 = Block(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 16, features * 4, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8, features * 2, down=False, act="relu", use_dropout=True
                         )
        self.up4 = Block(
            features * 4, features * 1, down=False, act="relu", use_dropout=False
        )
        self.out_tr = nn.Sequential(
            nn.Conv3d(features * 2, features, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(features),
            nn.LeakyReLU(0.2),
            nn.Conv3d(features, in_channels, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d0 = self.in_tr(x)
        d1 = self.initial_down(d0)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)

        d4 = self.bottleneck1(d4)
        d4 = self.bottleneck2(d4)
        d4 = self.bottleneck3(d4)

        up1 = self.up1(d4)
        up2 = self.up2(torch.cat([up1, d3], dim=1))
        up3 = self.up3(torch.cat([up2, d2], dim=1))
        up4 = self.up4(torch.cat([up3, d1], dim=1))
        up0 = self.out_tr(torch.cat([up4, d0], dim=1))

        return up0


class SPADEGenerator(nn.Module):
    # TODO: SPADE spatial adaptive denormalization used in decoder block
    def __init__(self, in_channels=1, features=16):
        super().__init__()
        self.in_tr = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=5, padding=2, stride=1),
            nn.InstanceNorm3d(features),
            nn.LeakyReLU(0.2),
        )
        self.initial_down = nn.Sequential(
            nn.Conv3d(features, features * 2, 4, 2, 1, padding_mode="reflect"),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 8, features * 16, down=True, act="leaky", use_dropout=False)

        self.bottleneck1 = bottleneck(features * 16, features * 16)
        self.bottleneck2 = bottleneck(features * 16, features * 16)
        self.bottleneck3 = bottleneck(features * 16, features * 16)

        self.up1 = Block(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.spade1 = SPADE(features * 16)
        self.up2 = Block(features * 16, features * 4, down=False, act="relu", use_dropout=True)
        self.spade2 = SPADE(features * 8)
        self.up3 = Block(features * 8, features * 2, down=False, act="relu", use_dropout=True)
        self.spade3 = SPADE(features * 4)
        self.up4 = Block(features * 4, features * 1, down=False, act="relu", use_dropout=False)
        self.spade4 = SPADE(features * 2)
        self.out_tr = nn.Sequential(
            nn.Conv3d(features * 2, features, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(features),
            nn.LeakyReLU(0.2),
            nn.Conv3d(features, in_channels, kernel_size=3, padding=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x, segmap):
        d0 = self.in_tr(x)
        d1 = self.initial_down(d0)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)

        d4 = self.bottleneck1(d4)
        d4 = self.bottleneck2(d4)
        d4 = self.bottleneck3(d4)

        up1 = self.up1(d4)
        spade1 = self.spade1(torch.cat([up1, d3], dim=1), segmap)
        up2 = self.up2(spade1)
        spade2 = self.spade2(torch.cat([up2, d2], dim=1), segmap)
        up3 = self.up3(spade2)
        spade3 = self.spade3(torch.cat([up3, d1], dim=1), segmap)
        up4 = self.up4(spade3)
        spade4 = self.spade4(torch.cat([up4, d0], dim=1), segmap)
        up0 = self.out_tr(spade4)
        return up0
