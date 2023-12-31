import os
import numpy as np
import imageio
from torch.utils.data.dataset import Dataset

class SUNRGBD(Dataset):

    def __init__(self, data_dir, transform=None, phase='train'):
        r"""SUNRGBD dataset.

        Initialize the SUNRGBD dataset.

        Args:
            data_dir: SUNRGBD dataset directory path, 
                      default is ../datasets/sunrgbd.
            phase: Dataset loading phase, train or test.
            transform: transform dataset to tenser, refer to preprocessing :see proprocessing.py
        """
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform

        try:
            tmp_file = os.path.join(self.data_dir, 'train_rgb.txt')
            self.img_dir_train = []
            self.img_dir_test = []
            with open(tmp_file, 'r') as f:
                train_imgs = f.read().splitlines()
                for img in train_imgs:
                    self.img_dir_train.append(f'{data_dir}/{img}')
            tmp_file = os.path.join(self.data_dir, 'test_rgb.txt')
            with open(tmp_file, 'r') as f:
                test_imgs = f.read().splitlines()
                for img in test_imgs:
                    self.img_dir_test.append(f'{data_dir}/{img}')

            tmp_file = os.path.join(self.data_dir, 'train_depth.txt')
            self.depth_dir_train = []
            self.depth_dir_test = []
            with open(tmp_file, 'r') as f:
                train_imgs = f.read().splitlines()
                for img in train_imgs:
                    self.depth_dir_train.append(f'{data_dir}/{img}')
            tmp_file = os.path.join(self.data_dir, 'test_depth.txt')
            with open(tmp_file, 'r') as f:
                test_imgs = f.read().splitlines()
                for img in test_imgs:
                    self.depth_dir_test.append(f'{data_dir}/{img}')

            tmp_file = os.path.join(self.data_dir, 'train_label.txt')
            self.label_dir_train = []
            self.label_dir_test = []
            with open(tmp_file, 'r') as f:
                train_imgs = f.read().splitlines()
                for img in train_imgs:
                    self.label_dir_train.append(f'{data_dir}/{img}')
            tmp_file = os.path.join(self.data_dir, 'test_label.txt')
            with open(tmp_file, 'r') as f:
                test_imgs = f.read().splitlines()
                for img in test_imgs:
                    self.label_dir_test.append(f'{data_dir}/{img}')

        except:
            raise IOError(f'{tmp_file} not found.')

    def __len__(self):
        if self.phase == 'train':
            return len(self.img_dir_train)
        elif self.phase == 'test':
            return len(self.img_dir_test)
        else:
            raise NotImplementedError('Only train and test phase are supported.')

    def __getitem__(self, index):
        if self.phase == 'train':
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        elif self.phase == 'test':
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test
        else:
            raise NotImplementedError('Only train and test phase are supported.')
        label = np.load(label_dir[index])
        depth = imageio.imread(depth_dir[index])
        rgb = imageio.imread(img_dir[index])
        # if transform is not None, preprocess the dataset
        if self.transform:
            sample = {'rgb': rgb,
                      'depth': depth,
                      'label': label}
            sample = self.transform(sample)
            return sample
        
        sample = {'rgb': rgb, 'depth': depth, 'label': label}
        
        return sample