import imageio
import os
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset

class NYUv2(Dataset):

    def __init__(self, data_dir, transform=None, phase='train'):
        r"""NYUv2 dataset.

        Initialize the NYUv2 dataset.

        Args:
            data_dir: NYUv2 dataset directory path, 
                      default is ../datasets/nyuv2.
            phase: Dataset loading phase, train or test.
            transform: transform dataset to tenser, refer to preprocessing :see proprocessing.py
        """
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
        label = imageio.imread(label_dir[index])
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