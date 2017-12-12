import os

import numpy as np
#import pandas as pd
#from skimage import io

import torch
from torch.utils.data import Dataset


class PositionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.positions = pd.read_csv(csv_file, sep=' ', header=0, index_col=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # img1 = os.path.join(self.image_dir, '{}.png'.format(self.positions.ix[idx, 0]))
        # img2 = os.path.join(self.image_dir, '{}.png'.format(self.positions.ix[idx, 0] + 1))
        img1 = os.path.join(self.image_dir, 'im_{:03}.png'.format(idx))
        img2 = os.path.join(self.image_dir, 'im_{:03}.png'.format(idx + 1))
        # Remove alpha channel with [:, :, :-1]
        # XXX Swap axes for channel first (3xHxW)
        #frame1 = io.imread(img1)[:, :, :-1].swapaxes(0, -1).astype(float)
        #frame2 = io.imread(img2)[:, :, :-1].swapaxes(0, -1).astype(float)
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)

        position = self.positions.ix[idx, 1:].as_matrix().astype('float')
        sample = {'frames': (frame1, frame2), 'position': position}

        return sample


def ang_diff(a, b):
    """ a and b are in radians """
    diff = (a - b) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff


class Env2DDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = np.loadtxt(data_file)
        self.transform = transform

    def __len__(self):
        return np.shape(self.data)[0] - 2

    def _extract_frame(self, idx):
        return np.expand_dims(self.data[idx, 3:], axis=0)

    def __getitem__(self, idx):
        # delta_p = self.data[idx + 1, 0:2] - self.data[idx, 0:2]
        delta_theta = ang_diff(self.data[idx + 1, 2], self.data[idx, 2])
        # delta_pose = np.hstack((delta_p, delta_theta))
        delta_pose = np.expand_dims(delta_theta, 1)

        frames = [self._extract_frame(idx), self._extract_frame(idx + 1),
                  self._extract_frame(idx + 2)]
        frames = [torch.Tensor(f) for f in frames]
        if self.transform:
            frames = [self.transform(f) for f in frames]

        return {'frames': frames, 'position': delta_pose}

class ThorDataset(Dataset):
  def __init__(self, directory, transform=None):
    self.data_angles = np.radians(np.load(os.path.join(directory, "data_angles.npy")))
    #print(self.data_angles)
    #print(int(self.data_angles.shape[0] / 3))
    self.data_images = np.load(os.path.join(directory, "data_images.npy"))
    self.transform = transform

  def __len__(self):
    return int(self.data_angles.shape[0] / 3)

  def _extract_frame(self, idx):
    #return np.expand_dims(np.rollaxis(self.data_images[idx], 2)[0], axis=0)
    return np.expand_dims(self.data_images[idx], axis=0)

  def __getitem__(self, idx):
    delta_theta = ang_diff(self.data_angles[3 * idx + 1][0], self.data_angles[3 * idx][0])
    delta_phi = ang_diff(self.data_angles[3 * idx + 1][1], self.data_angles[3 * idx][1])
    delta_pose = np.array([delta_theta, delta_phi])
#np.expand_dims(delta_theta, 1)##.append(delta_phi)
    #print(delta_pose)





    frames = [self._extract_frame(3 * idx), self._extract_frame(3 * idx + 1), self._extract_frame(3 * idx + 2)]
    frames = [torch.Tensor(f) for f in frames]
    if self.transform:
      frames = [self.transform(f) for f in frames]

    return {'frames' : frames, 'position' : delta_pose}
