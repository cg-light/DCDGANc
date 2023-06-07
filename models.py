# -*- coding: utf-8 -*-
"""
@ Project Name: styleVAEGAN
@ Author: Jing
@ TIME: 13:17/15/11/2021
"""
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class AdaIN(nn.Module):
    def __init__(self, weight_dim, channels):
        super(AdaIN, self).__init__()
        # print(channels)
        self.weight = nn.Linear(weight_dim, 2 * channels)

    def forward(self, x, w):
        eps = 0.0001
        mu = torch.mean(torch.mean(x, dim=2, keepdim=True), dim=3, keepdim=True)
        mu = mu.repeat(1, 1, x.shape[2], x.shape[3])
        seta = torch.mean(torch.mean((x - mu) ** 2, dim=2, keepdim=True), dim=3, keepdim=True).sqrt() + eps
        # print(x.shape)
        y = self.weight(w)
        # print(y.shape)
        y = y.view(2, x.shape[0], x.shape[1], 1, 1)
        out = y[0] * (x - mu) / seta + y[1]
        # print(out.shape)
        return out


def norm(x, gama, beta):
    mu = torch.mean(x, dim=1, keepdim=True)
    seta = torch.mean((x ** 2 - mu ** 2), dim=1, keepdim=True).sqrt()
    out = gama * (x - mu) / seta + beta
    return out


class SPADE(nn.Module):
    def __init__(self, in_size, out_channels):
        super(SPADE, self).__init__()

        self.fc = nn.Linear(in_size[0], in_size[2] * in_size[3])
        self.in_size = in_size
        self.spade1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.spade_gama = nn.Conv2d(3, out_channels, 3, 1, 1, bias=False)
        self.spade_beta = nn.Conv2d(3, out_channels, 3, 1, 1, bias=False)

    def forward(self, x, z):
        z = self.fc(z).view(z.size(0), 1, self.in_size[2], self.in_size[3])
        z = self.spade1(z)
        gama = self.spade_gama(z)
        beta = self.spade_beta(z)
        out = norm(x, gama, beta)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_c, norm_type, noise=True):
        super(ResidualBlock, self).__init__()

        in_features = in_size[1]
        if in_features == out_c:
            self.use_1x1 = False
        else:
            self.use_1x1 = True
        self.norm = norm_type
        self.noise = noise

        self.conv1 = nn.Conv2d(in_features, out_c, 3, stride=1, padding=1, groups=1, bias=False)
        if norm_type == 'spade':
            norm_layer = SPADE(in_size, out_c)
        elif norm_type == 'adain':
            norm_layer = AdaIN(in_size[0], out_c)
        else:
            norm_layer = nn.InstanceNorm2d(out_c, 0.8, track_running_stats=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, groups=1, bias=False)
        if self.use_1x1:
            self.conv3 = nn.Conv2d(in_features, out_c, 1)
        self.norm_layer1 = norm_layer
        self.norm_layer2 = norm_layer
        if noise:
            self.insert_noise = nn.Linear(1, out_c)

    def forward(self, x, z):
        if self.noise:
            noise = torch.rand(x.shape[0], x.shape[2], x.shape[3], 1).cuda()
            noise = self.insert_noise(noise).permute((0, 3, 2, 1))
            x = x + noise
        # print(noise.shape, x.shape)
        x_out = self.conv1(x)
        # print(x_out.shape, noise.shape)
        if self.norm == 'spade' or self.norm == 'adain':
            x_out = self.norm_layer1(x_out, z)
        else:
            x_out = self.norm_layer1(x_out)
        x_out = self.act1(x_out)
        x_out = self.conv2(x_out)
        if self.norm == 'spade' or self.norm == 'adain':
            x_out = self.norm_layer2(x_out, z)
        else:
            x_out = self.norm_layer2(x)
        if self.use_1x1:
            return self.conv3(x) + x_out
        else:
            return x + x_out


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels, weight_dim, h, w, norm_type, noise=True):
        super(down_block, self).__init__()
        self.norm_type = norm_type
        self.noise = noise
        group_num = 1

        model = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 4, 2, 1, groups=group_num)])
        if norm_type == 'adain':
            model.extend([AdaIN(weight_dim, out_channels), nn.LeakyReLU(0.2, inplace=True)])
        elif norm_type == 'spade':
            model.extend([SPADE([weight_dim, out_channels, h, w], out_channels), nn.LeakyReLU(0.2, inplace=True)])
        elif norm_type == 'instance':
            model.extend([nn.InstanceNorm2d(out_channels, 0.8), nn.LeakyReLU(0.2, inplace=True)])
        elif not norm_type:
            model.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            print('choose appropriate normalization')

        self.models = model
        if noise:
            self.insert_noise = nn.Linear(1, in_channels)

    def forward(self, x, z):
        if self.noise:
            noise = torch.rand(x.shape[0], x.shape[2], x.shape[3], 1).cuda()
            noise = self.insert_noise(noise).permute((0, 3, 2, 1))
            x = x + noise
        i = 0
        for model in self.models:
            if self.norm_type == 'spade' or self.norm_type == 'adain':
                if i == 1:
                    # print(model, i)
                    x = model(x, z)
                else:
                    x = model(x)
            else:
                x = model(x)
            i += 1
        return x


class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, weight_dim, h, w, norm_type, noise=True):
        super(up_block, self).__init__()
        self.norm_type = norm_type
        self.noise = noise
        group_num = 1

        model = nn.ModuleList([nn.Upsample(scale_factor=2),
                               nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group_num)])
        if norm_type == 'adain':
            model.extend([AdaIN(weight_dim, out_channels), nn.LeakyReLU(0.2, inplace=True)])
        elif norm_type == 'spade':
            model.extend([SPADE([weight_dim, out_channels, h, w], out_channels), nn.LeakyReLU(0.2, inplace=True)])
        elif norm_type == 'instance':
            model.extend([nn.InstanceNorm2d(out_channels, 0.8), nn.LeakyReLU(0.2, inplace=True)])
        elif not norm_type:
            model.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            print('choose appropriate normalization')

        self.models = model
        if noise:
            self.insert_noise = nn.Linear(1, in_channels)

    def forward(self, x, z):
        if self.noise:
            noise = torch.rand(x.shape[0], x.shape[2], x.shape[3], 1).cuda()
            noise = self.insert_noise(noise).permute((0, 3, 2, 1))
            x = x + noise
        i = 0
        for model in self.models:
            if self.norm_type == 'spade' or self.norm_type == 'adain':
                if i == 2:
                    x = model(x, z)
                else:

                    x = model(x)
            else:
                x = model(x)
            i += 1
        return x


class Res_Generator(nn.Module):
    def __init__(self, weight_dim, in_size, norm_type, res_num=6, noise=False):
        super(Res_Generator, self).__init__()
        channel, h, w = in_size
        self.c, self.h, self.w = channel, h, w
        self.norm = norm_type

        # down sample
        h = h // 2
        w = w // 2
        curr_channel = 32
        if norm_type == 'instance':
            self.in_layer = nn.Sequential(nn.Linear(weight_dim, h * w * 4, bias=True))
            model = nn.ModuleList([down_block(channel + 1, curr_channel, weight_dim, h, w, False, noise)])
        else:
            model = nn.ModuleList([down_block(channel, curr_channel, weight_dim, h, w, False, noise)])
        for _ in range(3):
            h = h // 2
            w = w // 2
            model.append(down_block(curr_channel, curr_channel * 2, weight_dim, h, w, norm_type, noise))
            curr_channel *= 2

        h = h // 2
        w = w // 2
        model.extend([down_block(curr_channel, curr_channel, weight_dim, h, w, norm_type, noise)])  # ,

        for i in range(res_num):
            model.append(ResidualBlock([weight_dim, curr_channel, h, w], curr_channel, norm_type, noise))  # 256

        # # up sample and output
        for _ in range(4):
            h *= 2
            w *= 2
            model.append(up_block(curr_channel, curr_channel // 2, weight_dim, h, w, norm_type, noise))
            curr_channel = curr_channel // 2

        self.models = model
        self.out_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(curr_channel, channel, 3, 1, 1),  # 16
            nn.Tanh())

    def forward(self, clabel, label, z):
        x = clabel.repeat(1, self.c * self.h * self.w).view(z.size(0), self.c, self.h, self.w)
        z = torch.cat((z, label), 1)
        if self.norm == 'instance':
            z = self.in_layer(z).view(x.size(0), 1, self.h, self.w)
            x = torch.cat((x, z), 1)
        for model in self.models:
            x = model(x, z)
        return self.out_layer(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Encoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 7, 2, 3, bias=False),
            nn.InstanceNorm2d(64, 0.8),
            nn.ReLU(inplace=True))
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(input_shape[1] * input_shape[2] // 4, latent_dim)
        self.fc_logvar = nn.Linear(input_shape[1] * input_shape[2] // 4, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)  # torch.Size([2, 64, 128, 128])
        out = self.pooling(out)  # torch.Size([2, 64, 16, 16])
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)  # torch.Size([2, 6])
        logvar = self.fc_logvar(out)
        return mu, logvar


class MulticlassDis(nn.Module):
    def __init__(self, img_shape, c_dim):
        super(MulticlassDis, self).__init__()
        channels, h, w = img_shape
        self.h = h
        self.w = w

        def discriminator_block(in_filters, out_filters, norm_type=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm_type:
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.in_layers = nn.ModuleList([nn.Linear(c_dim, h * w),
                                       nn.Linear(c_dim, h * w // 4),
                                       nn.Linear(c_dim, h * w // 16)])
        self.models = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                'class %s' % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 256)
                )
            )
            self.out_layers.append(nn.Conv2d(256, 1, 3, padding=1))
        self.down_sample = nn.AvgPool2d(channels + 1, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, img, label, gt, y):
        loss_c = sum([torch.mean(out - gt) ** 2 for out in self.forward(img, label)])
        return loss_c, True

    def forward(self, img, label):
        # input = label.repeat(1, self.img_size * self.img_size).view(img.size(0), 1, self.img_size, self.img_size)
        outputs = []
        i = 0
        for model in self.models:
            h, w = self.h // 2 ** (i + 4), self.w // 2 ** (i + 4)
            label_in = self.in_layers[i](label).view(img.size(0), -1, h, w)
            feature = model(img)
            outputs.append(self.out_layers[i](feature * label_in))
            img = self.down_sample(img)
            i += 1
        return outputs


class Multiclass(nn.Module):
    def __init__(self, img_shape, c_dim):
        super(Multiclass, self).__init__()
        channels, h, w = img_shape
        self.h = h
        self.w = w

        def discriminator_block(in_filters, out_filters, norm_type=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm_type:
                layers.append(nn.InstanceNorm2d(out_filters))
                # layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # self.in_layer = nn.ModuleList([nn.Linear(c_dim, h * w),
        #                                nn.Linear(c_dim, h * w // 4),
        #                                nn.Linear(c_dim, h * w // 16)])
        self.models = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                'class %s' % i,
                nn.Sequential(
                    *discriminator_block(channels, 64),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 256),
                    nn.Conv2d(256, 1, 3, padding=1)
                )
            )
        self.down_sample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, img, label, gt, clabel):
        loss_c = sum([torch.mean(out - gt) ** 2 for out in self.forward(img, label, clabel)])
        return loss_c, True

    def forward(self, img, label, clabel):
        outputs = []
        i = 0
        for model in self.models:
            outputs.append(model(img))
            img = self.down_sample(img)
            i += 1
        return outputs