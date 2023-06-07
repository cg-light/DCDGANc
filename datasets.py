# -*- coding: utf-8 -*-
"""
@ Project Name: styleVAEGAN
@ Author: Jing
@ TIME: 13:17/15/11/2021
"""
import glob
import random
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, c_dim, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files = {}
        path = os.path.join(root, "%s" % mode)
        self.class_count = 0
        for parent, dirnames, _ in os.walk(path):
            for dirname in dirnames:
                # print(dirname)
                files = sorted(glob.glob(os.path.join(parent, dirname) + "/*.*"))
                self.files[self.class_count] = files
                self.class_count += 1
                if self.class_count == c_dim:
                    break
            break
        print(self.class_count)

    def __getitem__(self, index):
        y = random.randint(0, (self.class_count - 1))
        if self.unaligned:
            image = Image.open(self.files[y][random.randint(0, (len(self.files[y]) - 1))])
        else:
            # for y in range(self.class_count):
            image = Image.open(self.files[y][index % len(self.files[y])])
        img = self.transform(image)
        # print(item)
        return img, y

    def __len__(self):
        l = 0
        for key in self.files.keys():
            l += len(self.files[key])  # if l < len(self.files[i]) else l
        return l
