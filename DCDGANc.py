# -*- ecoding: utf-8 -*-
# @ModuleName: BicycleGAN
# @ProjectName: BicycleGAN 
# @Author: Kilo
# @Time: 2021/8/20-21:30
import argparse
import datetime
import time
import sys

from torch import autograd
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from models import *
from datasets import *

import torch

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="../../../paras/DCDGANc", help="storage path")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_class", type=int, default=5, help="number of classes")
parser.add_argument("--dataset_name", type=str, default="carpet", help="name of the dataset")
parser.add_argument("--noise", type=str, default=False, help="insert noise or not")
parser.add_argument("--sample_interval", type=int, default=250, help="interval between saving generator samples")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--res_num", type=int, default=4, help="number of classes")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between model checkpoints")
parser.add_argument("--norm_type", type=str, default="spade", help="name of the dataset")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=1, help="latent loss weight")
parser.add_argument("--lambda_gp", type=float, default=2.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
parser.add_argument("--lambda_cls", type=float, default=5, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)
os.makedirs("%s/images/%s" % (opt.filename, opt.dataset_name), exist_ok=True)
os.makedirs("%s/saved_models/%s" % (opt.filename, opt.dataset_name), exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
mae_loss = torch.nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = Res_Generator(opt.latent_dim + opt.n_class, input_shape, opt.norm_type, opt.res_num, noise=opt.noise)
encoder = Encoder(opt.latent_dim, input_shape)
D_VAE = Multiclass(input_shape, opt.n_class)

if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    mae_loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(
        torch.load("%s/saved_models/%s/generator_%d.pth" % (opt.filename, opt.dataset_name, opt.epoch)))
    encoder.load_state_dict(
        torch.load("%s/saved_models/%s/encoder_%d.pth" % (opt.filename, opt.dataset_name, opt.epoch)))
    D_VAE.load_state_dict(torch.load("%s/saved_models/%s/D_VAE_%d.pth" % (opt.filename, opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.Resize((int(1.12 * opt.img_height), int(1.12 * opt.img_width)), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]

dataloader = DataLoader(
    ImageDataset("../../../datasets/%s" % opt.dataset_name, opt.n_class, transforms_=transforms_, mode='train',
                 unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
print(len(dataloader))


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    generator.eval()
    img_samples = None
    for label in range(opt.n_class):
        real_label = torch.zeros((1, opt.n_class))
        real_label[0, label] = 1
        real_label = real_label.repeat(8, 1)
        real_label = real_label.type(Tensor)
        clabels = 2 * label / (opt.n_class - 1) - 1
        clabels = np.array(clabels)
        clabels = Tensor(clabels).repeat(8, 1)  # -1, 1
        # Sample latent representations
        sampled_z = Tensor(np.random.normal(0, 1, (8, opt.latent_dim)))  # * torch.exp(logvar) + mu)
        # Generate samples
        fake_B = generator(clabels, real_label, sampled_z)
        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B.data.cpu()], 2)
        img_sample = fake_B
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "%s/images/%s/%s.png" % (opt.filename, opt.dataset_name, batches_done), nrow=8,
               normalize=True)
    generator.train()


def compute_gradient_penalty(D, real_samples, fake_samples, l):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D.compute_loss(interpolates, l, valid)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        # grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# 重参数
def reparameterization(mu, lovar):
    std = torch.exp(lovar / 2)
    sample_z = Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim)))
    z = sample_z * std + mu
    return z


def embedding(condition, n=opt.n_class):
    label = torch.zeros(n)
    label[condition] = 1
    return label


# ----------
#  Training
# ----------
# Adversarial loss
valid = 1
fake = 0

prev_time = time.time()
print("Training...\n")
if __name__ == "__main__":
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, (imgs, ys) in enumerate(dataloader):

            # set model input
            real_B = imgs.type(Tensor)
            y = ys
            labels = torch.zeros((real_B.size(0), opt.n_class))
            j = 0
            for yc in ys:
                labels[j, yc] = 1
                j += 1
            labels = labels.type(Tensor)
            # glabels = glabels.type(Tensor)
            clabels = ((2 * ys / (opt.n_class - 1) - 1)).type(Tensor)  # -1, 1
            # ys = ys.type(torch.LongTensor).cuda()

            # -------------------------------
            #  Train Generator and Encoder
            # -------------------------------
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()

            # ----------
            # cVAE-GAN 编码生成的图像
            # ----------

            # Produce output using encoding of B (cVAE-GAN)
            mu, logvar = encoder(real_B)
            # Kullback-Leibler divergence of encoded B
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            encoder_z = reparameterization(mu, logvar)
            fake_B = generator(clabels, labels, encoder_z)
            # Pixelwise loss of translated image by VAE
            loss_pixel = mae_loss(fake_B, real_B)
            # Adversarial loss
            loss_VAE_GAN = D_VAE.compute_loss(fake_B, labels, valid)

            # ---------
            # cLR-GAN 采样z生成的图像骗过鉴别器
            # ---------
            # Produce output using sampled z (cLR-GAN)
            sampled_z = Tensor(np.random.normal(0, 1, (real_B.size(0), opt.latent_dim)))
            _fake_B = generator(clabels, labels, sampled_z)
            loss_LR_GAN = D_VAE.compute_loss(_fake_B, labels, valid)

            # ----------------------------------
            # Total Loss (Generator + Encoder)
            # ----------------------------------
            loss_GAN = loss_VAE_GAN + loss_LR_GAN
            loss_GE = loss_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl

            loss_GE.backward(retain_graph=True)
            optimizer_E.step()

            # ---------------------
            # Generator Only Loss 生成图像和潜在空间采样的向量距离最小
            # ---------------------

            # Latent L1 loss
            _mu, _ = encoder(_fake_B)
            loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z)
            loss_latent.backward()
            optimizer_G.step()

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)真实图像和编码向量生成的图像损失
            # ----------------------------------

            optimizer_D_VAE.zero_grad()

            loss_adv_r = D_VAE.compute_loss(real_B, labels, valid)
            loss_adv_f = D_VAE.compute_loss(fake_B.detach(), labels, fake)
            loss_adv_f_ = D_VAE.compute_loss(_fake_B.detach(), labels, fake)
            loss_VAE_gp = opt.lambda_gp * (compute_gradient_penalty(D_VAE, real_B.detach(), _fake_B.detach(), labels) +
                                           compute_gradient_penalty(D_VAE, real_B.detach(), fake_B.detach(), labels))
            loss_D_VAE = 2 * loss_adv_r + loss_adv_f + loss_adv_f_ + loss_VAE_gp
            loss_D_VAE.backward()
            optimizer_D_VAE.step()

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:  # and epoch > 100:
            # Save model checkpoints
            torch.save(generator.state_dict(),
                       "%s/saved_models/%s/generator_%d.pth" % (opt.filename, opt.dataset_name, epoch))
            torch.save(encoder.state_dict(),
                       "%s/saved_models/%s/encoder_%d.pth" % (opt.filename, opt.dataset_name, epoch))
            torch.save(D_VAE.state_dict(), "%s/saved_models/%s/D_VAE_%d.pth" % (opt.filename, opt.dataset_name, epoch))

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D adv_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s\n"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D_VAE.item(),
                (loss_VAE_GAN + loss_LR_GAN).item(),
                (opt.lambda_pixel * loss_pixel).item(),
                (opt.lambda_kl * loss_kl).item(),
                (opt.lambda_latent * loss_latent).item(),
                time_left,
            )
        )
