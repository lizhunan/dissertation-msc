import imageio
import numpy as np
import cv2
import os
from torch import from_numpy
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset

class NYUv2(Dataset):

    def __init__(self, transform=None, phase=True, data_dir=os.path.expanduser('~/dissertation-msc/datasets/nyuv2')):
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform

        try:
            tmp_file = os.path.join(self.data_dir, 'train.txt')
            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []
            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []
            with open(tmp_file, 'r') as f:
                train_imgs = f.read().splitlines()
                for img in train_imgs:
                    self.img_dir_train.append(f'{data_dir}/train/rgb/{img}.png')
                    self.depth_dir_train.append(f'{data_dir}/train/depth/{img}.png')
                    self.label_dir_train.append(f'{data_dir}/train/labels_40/{img}.png')
            tmp_file = os.path.join(self.data_dir, 'test.txt')
            with open(tmp_file, 'r') as f:
                test_imgs = f.read().splitlines()
                for img in test_imgs:
                    self.img_dir_test.append(f'{data_dir}/test/rgb/{img}.png')
                    self.depth_dir_test.append(f'{data_dir}/test/depth/{img}.png')
                    self.label_dir_test.append(f'{data_dir}/test/labels_40/{img}.png')
        except:
            raise IOError(f'{tmp_file} not found.')
        
    def __len__(self):
        if self.phase:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)
    
    def __getitem__(self, index):

        if self.phase:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test
        # rgb = self._load_img(img_dir[index])
        # depth = self._load_img(depth_dir[index])
        # label = self._load_img(label_dir[index])
        label = imageio.imread(label_dir[index])
        depth = imageio.imread(depth_dir[index])
        rgb = imageio.imread(img_dir[index])
        
        if self.transform:
            sample = {'rgb': rgb,
                      'depth': depth,
                      'label': label}
            sample = self.transform(sample)
            return sample
        
        sample = {'rgb': rgb, 'depth': depth, 'label': label}

        return sample
    
    def _load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(img.dtype)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img