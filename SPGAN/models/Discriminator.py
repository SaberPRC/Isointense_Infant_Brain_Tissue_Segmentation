import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class PatchGANDiscriminator(nn.Module):
    # TODO: discriminator for PatchGAN without condition input
    def __init__(self, input_nc, ndf=32, n_layers=3, use_sigmoid=False):
        super(PatchGANDiscriminator, self).__init__()
        model = [
            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(ndf),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv3d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm3d(ndf*nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        model += [
            nn.Conv3d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(ndf*nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        model += [nn.Conv3d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        if use_sigmoid:
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    # TODO: PatchGAN discriminator with condition input
    def __init__(self, in_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv3d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x


if __name__ == '__main__':
    # TODO: test for patch gan discriminator
    from IPython import embed
    model = PatchGANDiscriminator(1)
    x = torch.randn(2, 1, 128, 128, 48)
    embed()