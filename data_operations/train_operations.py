import torch
from PIL import Image
import SimpleITK as sitk
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
from data_operations.augmentation import *
from config import config


class ImageDataset(data.Dataset):
    def __init__(self, images, groundtruth, img_size, augment):
        self.images = images
        self.gt = groundtruth
        self.augment = augment
        self.size = len(images)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size))])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.gt[idx]

        img = sitk.ReadImage(x)
        curr = sitk.GetArrayFromImage(img)
        im = Image.fromarray((curr))


        x = self.transform(im)
        x = np.array(x)


        if self.augment:
            x = x[np.newaxis, ...]
            x = apply_aug(x)
            x = x[0]

        if config.num_channels == 1:
            if x.shape[-1] == 3:
                x = x[:, :, 0]
            x = x[..., np.newaxis]


        if config.normalize:
            x_max = np.max(x)
            x_min = np.min(x)
            x = (x - x_min) / (x_max - x_min)
            #x = x/ 255

        if config.z_score:
            x_mean = np.mean(x)
            x_std = np.std(x)
            x = (x - x_mean) / x_std



        x = x.astype(np.float32)
        x = np.swapaxes(x, 0, -1)
        y = np.array(y)
        return x, y
