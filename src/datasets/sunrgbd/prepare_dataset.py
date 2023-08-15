import os
from tqdm import tqdm
import urllib.request
from zipfile import ZipFile
import h5py
import numpy as np
import scipy.io

# see: http://rgbd.cs.princeton.edu/ in section Data and Annotation
RGBD_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
TOOLBOX_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip'

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class Preparation:

    def __init__(self, args):
        self.args = args
        self.output_path = os.path.expanduser(self.args.rgbd_output_path)
        self.SUNRGBD = os.path.join(os.path.dirname(__file__), 'SUNRGBD.zip')
        self.SUNRGBDtoolbox = os.path.join(os.path.dirname(__file__), 'SUNRGBDtoolbox.zip')

    def __call__(self):
        if self.args.download:
            self._download()
        self._rgbd()

    def _download(self):
        with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=RGBD_URL.split('/')[-1]) as t:
            urllib.request.urlretrieve(RGBD_URL,
                                    filename=self.SUNRGBD, 
                                    reporthook=t.update_to)
        with DownloadProgressBar(unit='B', unit_scale=True,
                            miniters=1, desc=TOOLBOX_URL.split('/')[-1]) as t:
            urllib.request.urlretrieve(TOOLBOX_URL,
                                    filename=self.SUNRGBDtoolbox, 
                                    reporthook=t.update_to)
            
    def _rgbd(self):
        
        print('Extract labels from SUNRGBD toolbox')
        with ZipFile(self.SUNRGBD, 'r') as zip_ref:
            zip_ref.extractall(path=os.path.join(self.output_path, ''))
        with ZipFile(self.SUNRGBDtoolbox, 'r') as zip_ref:
            zip_ref.extractall(path=os.path.join(self.output_path, ''))

        # remove downloaded file
        if self.SUNRGBD is not None:
            print(f"Removing downloaded zip file: `{self.SUNRGBD}`")
            os.remove(self.SUNRGBD)

        if self.SUNRGBDtoolbox is not None:
            print(f"Removing downloaded zip file: `{self.SUNRGBDtoolbox}`")
            os.remove(self.SUNRGBDtoolbox)

        SUNRGBDMeta_dir = os.path.join(os.path.join(self.output_path, 'SUNRGBDtoolbox'), 'Metadata/SUNRGBDMeta.mat')
        allsplit_dir = os.path.join(os.path.join(self.output_path, 'SUNRGBDtoolbox'), 'traintestSUNRGBD/allsplit.mat')
        SUNRGBD2Dseg_dir = os.path.join(os.path.join(self.output_path, 'SUNRGBDtoolbox'), 'Metadata/SUNRGBD2Dseg.mat')
        img_dir_train = []
        depth_dir_train = []
        label_dir_train = []
        img_dir_test = []
        depth_dir_test = []
        label_dir_test = []

        SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')

        # load the data from the matlab file
        SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
                                    struct_as_record=False)['SUNRGBDMeta']
        split = scipy.io.loadmat(allsplit_dir, squeeze_me=True,
                                struct_as_record=False)
        split_train = split['alltrain']

        seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']
        
        for i, meta in tqdm(enumerate(SUNRGBDMeta)):
            meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
            real_dir = meta_dir.split('/n/fs/sun3d/data/SUNRGBD/')[1]
            depth_bfx_path = os.path.join(real_dir, 'depth_bfx/' + meta.depthname)
            rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

            label_path = os.path.join(real_dir, 'label/label.npy')
            label_path_full = os.path.join(self.output_path, 'SUNRGBD', label_path)

            # save segmentation (label_path) as numpy array
            if not os.path.exists(label_path_full):
                os.makedirs(os.path.dirname(label_path_full), exist_ok=True)
                label = np.array(
                    SUNRGBD2Dseg[seglabel[i][0]][:].transpose(1, 0)).\
                    astype(np.uint8)
                np.save(label_path_full, label)

            if meta_dir in split_train:
                img_dir_train.append(os.path.join('SUNRGBD', rgb_path))
                depth_dir_train.append(os.path.join('SUNRGBD', depth_bfx_path))
                label_dir_train.append(os.path.join('SUNRGBD', label_path))
            else:
                img_dir_test.append(os.path.join('SUNRGBD', rgb_path))
                depth_dir_test.append(os.path.join('SUNRGBD', depth_bfx_path))
                label_dir_test.append(os.path.join('SUNRGBD', label_path))

        # write file lists
        def _write_list_to_file(list_, filepath):
            with open(os.path.join(self.output_path, filepath), 'w') as f:
                f.write('\n'.join(list_))
            print('written file {}'.format(filepath))

        _write_list_to_file(img_dir_train, 'train_rgb.txt')
        _write_list_to_file(depth_dir_train, 'train_depth.txt')
        _write_list_to_file(label_dir_train, 'train_label.txt')
        _write_list_to_file(img_dir_test, 'test_rgb.txt')
        _write_list_to_file(depth_dir_test, 'test_depth.txt')
        _write_list_to_file(label_dir_test, 'test_label.txt')