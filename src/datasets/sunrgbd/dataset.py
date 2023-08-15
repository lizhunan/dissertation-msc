import os
import numpy as np
import imageio
from torch.utils.data.dataset import Dataset

class SUNRGBD(Dataset):

    def __init__(self, transform=None, phase=True, data_dir=os.path.expanduser('~/dissertation-msc/datasets/sunrgbd')):
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform

        try:
            tmp_file = os.path.join(self.data_dir, 'train_rgb.txt')
            with open(tmp_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            tmp_file = os.path.join(self.data_dir, 'train_depth.txt')
            with open(tmp_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            tmp_file = os.path.join(self.data_dir, 'train_label.txt')
            with open(tmp_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()
            tmp_file = os.path.join(self.data_dir, 'test_rgb.txt')
            with open(tmp_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            tmp_file = os.path.join(self.data_dir, 'test_depth.txt')
            with open(tmp_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            tmp_file = os.path.join(self.data_dir, 'test_label.txt')
            with open(tmp_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
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
        
        label = np.load(f'datasets/sunrgbd/{label_dir[index]}')
        depth = imageio.imread(f'datasets/sunrgbd/{depth_dir[index]}')
        image = imageio.imread(f'datasets/sunrgbd/{img_dir[index]}')
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        
        sample = {'image': image, 'depth': depth, 'label': label}
        
        return sample