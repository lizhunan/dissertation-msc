import os
import urllib.request
import numpy as np
from src.datasets.nyuv2.nyuv2 import NYUv2Base
import cv2
from tqdm import tqdm
from PIL import Image
import h5py
from scipy.io import loadmat

# https://github.com/VainF/nyuv2-python-toolkit/blob/master/splits.mat
NYUV2_SPLITS_FILEPATH = os.path.join(os.path.dirname(__file__),
                               'splits.mat')

# https://github.com/VainF/nyuv2-python-toolkit/blob/master/class13Mapping.mat
NYUV2_CLASSES_13_FILEPATH = os.path.join(os.path.dirname(__file__),
                                  'class13Mapping.mat')

# https://github.com/VainF/nyuv2-python-toolkit/blob/master/classMapping40.mat
NYUV2_CLASSES_40_FILEPATH = os.path.join(os.path.dirname(__file__),
                                  'classMapping40.mat')

# see: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
NYUV2_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'

class Preparation:

    def __init__(self, args):
       self.args = args

    def __call__(self):
        self._download()
        self._nyuv2()

    def _download(self):
        
        if self.args.download:
            # downloading nyu_depth_v2_labeled.mat
            with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=NYUV2_URL.split('/')[-1]) as t:
                urllib.request.urlretrieve(NYUV2_URL,
                                        filename=os.path.join(os.path.dirname(__file__), 'nyu_depth_v2_labeled.mat'), 
                                        reporthook=t.update_to)
    
    def _nyuv2(self):

        mat_path = os.path.join(os.path.dirname(__file__), 'nyu_depth_v2_labeled.mat')
        output_path = os.path.expanduser(self.args.nyuv2_output_path)
        
        # load mat file and extract images
        print(f'Loading mat file: {mat_path}')
        with h5py.File(f'{mat_path}', 'r') as f:
            rgb_images = np.array(f['images'])
            labels = np.array(f['labels'])
            depth_images = np.array(f['depths'])
            raw_depth_images = np.array(f['rawDepths'])

        # dimshuffle images
        rgb_images = self._dimshuffle(rgb_images, 'bc10', 'b01c')
        labels = self._dimshuffle(labels, 'b10', 'b01')
        depth_images = self._dimshuffle(depth_images, 'b10', 'b01')
        raw_depth_images = self._dimshuffle(raw_depth_images, 'b10', 'b01')

        # convert depth images (m to mm)
        depth_images = (depth_images * 1e3).astype('uint16')
        raw_depth_images = (raw_depth_images * 1e3).astype('uint16')

        # load split file (note that returned indexes start from 1)
        splits = loadmat(NYUV2_SPLITS_FILEPATH)
        train_idxs, test_idxs = splits['trainNdxs'][:, 0], splits['testNdxs'][:, 0]

        # load classes and class mappings (number of classes are without void)
        classes_40 = loadmat(NYUV2_CLASSES_40_FILEPATH)
        classes_13 = loadmat(NYUV2_CLASSES_13_FILEPATH)['classMapping13'][0][0]
        mapping_894_to_40 = np.concatenate([[0], classes_40['mapClass'][0]])
        mapping_40_to_13 = np.concatenate([[0], classes_13[0][0]])

        # get color (1 (void) + n_colors)
        colors = {
            894: np.array(NYUv2Base.CLASS_COLORS_894, dtype='uint8'),
            40: np.array(NYUv2Base.CLASS_COLORS_40, dtype='uint8'),
            13: np.array(NYUv2Base.CLASS_COLORS_13, dtype='uint8')
        }

        # save images
        for idxs, set_ in zip([train_idxs, test_idxs], ['train', 'test']):
            print(f"Processing set: {set_}")
            set_dir = NYUv2Base.SPLIT_DIRS[set_]
            rgb_base_path = os.path.join(output_path, set_dir, NYUv2Base.RGB_DIR)
            depth_base_path = os.path.join(output_path, set_dir,
                                        NYUv2Base.DEPTH_DIR)
            depth_raw_base_path = os.path.join(output_path, set_dir,
                                            NYUv2Base.DEPTH_RAW_DIR)
            labels_894_base_path = os.path.join(output_path, set_dir,
                                                NYUv2Base.LABELS_DIR_FMT.format(894))
            labels_40_base_path = os.path.join(
                output_path, set_dir, NYUv2Base.LABELS_DIR_FMT.format(40))
            labels_13_base_path = os.path.join(
                output_path, set_dir, NYUv2Base.LABELS_DIR_FMT.format(13))
            labels_894_colored_base_path = os.path.join(
                output_path, set_dir, NYUv2Base.LABELS_COLORED_DIR_FMT.format(894))
            labels_40_colored_base_path = os.path.join(
                output_path, set_dir, NYUv2Base.LABELS_COLORED_DIR_FMT.format(40))
            labels_13_colored_base_path = os.path.join(
                output_path, set_dir, NYUv2Base.LABELS_COLORED_DIR_FMT.format(13))

            os.makedirs(rgb_base_path, exist_ok=True)
            os.makedirs(depth_base_path, exist_ok=True)
            os.makedirs(depth_raw_base_path, exist_ok=True)
            os.makedirs(labels_894_base_path, exist_ok=True)
            os.makedirs(labels_13_base_path, exist_ok=True)
            os.makedirs(labels_40_base_path, exist_ok=True)
            os.makedirs(labels_894_colored_base_path, exist_ok=True)
            os.makedirs(labels_13_colored_base_path, exist_ok=True)
            os.makedirs(labels_40_colored_base_path, exist_ok=True)
            for idx in tqdm(idxs):
                # convert index from Matlab to [REST OF WORLD]
                idx_ = idx - 1

                # rgb image
                cv2.imwrite(os.path.join(rgb_base_path, f'{idx:04d}.png'),
                            cv2.cvtColor(rgb_images[idx_], cv2.COLOR_RGB2BGR))

                # depth image
                cv2.imwrite(os.path.join(depth_base_path, f'{idx:04d}.png'),
                            depth_images[idx_])

                # raw depth image
                cv2.imwrite(os.path.join(depth_raw_base_path, f'{idx:04d}.png'),
                            raw_depth_images[idx_])

                # label with 1+894 classes
                label_894 = labels[idx_]
                cv2.imwrite(os.path.join(labels_894_base_path, f'{idx:04d}.png'),
                            label_894)

                # colored label image
                # (normal png16 as this type does not support indexed palettes)
                label_894_colored = colors[894][label_894]
                cv2.imwrite(os.path.join(labels_894_colored_base_path,
                                        f'{idx:04d}.png'),
                            cv2.cvtColor(label_894_colored, cv2.COLOR_RGB2BGR))

                # label with 1+40 classes
                label_40 = mapping_894_to_40[label_894].astype('uint8')
                cv2.imwrite(os.path.join(labels_40_base_path, f'{idx:04d}.png'),
                            label_40)
                # colored label image
                # (indexed png8 with color palette)
                self._save_indexed_png(os.path.join(labels_40_colored_base_path,
                                            f'{idx:04d}.png'),
                                label_40, colors[40])

                # label with 1+13 classes
                label_13 = mapping_40_to_13[label_40].astype('uint8')
                cv2.imwrite(os.path.join(labels_13_base_path, f'{idx:04d}.png'),
                            label_13)
                
                # colored label image
                # (indexed png8 with color palette)
                self._save_indexed_png(os.path.join(labels_13_colored_base_path,
                                            f'{idx:04d}.png'),
                                label_13, colors[13])
        
        # save meta files
        print("Writing meta files")
        np.savetxt(os.path.join(output_path, 'class_names_1+13.txt'),
                NYUv2Base.CLASS_NAMES_13,
                delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_path, 'class_colors_1+13.txt'),
                NYUv2Base.CLASS_COLORS_13,
                delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_path, 'class_names_1+40.txt'),
                NYUv2Base.CLASS_NAMES_40,
                delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_path, 'class_colors_1+40.txt'),
                NYUv2Base.CLASS_COLORS_40,
                delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_path, 'class_names_1+894.txt'),
                NYUv2Base.CLASS_NAMES_894,
                delimiter=',', fmt='%s')
        np.savetxt(os.path.join(output_path, 'class_colors_1+894.txt'),
                NYUv2Base.CLASS_COLORS_894,
                delimiter=',', fmt='%s')

        # splits
        np.savetxt(os.path.join(output_path,
                                NYUv2Base.SPLIT_FILELIST_FILENAMES['train']),
                train_idxs,
                fmt='%04d')
        np.savetxt(os.path.join(output_path,
                                NYUv2Base.SPLIT_FILELIST_FILENAMES['test']),
                test_idxs,
                fmt='%04d')
        
        # remove downloaded file
        if mat_path is not None:
            print(f"Removing downloaded mat file: `{mat_path}`")
            os.remove(mat_path)

    def _save_indexed_png(self, filepath, label, colormap):
        # note that OpenCV is not able to handle indexed pngs correctly.
        img = Image.fromarray(np.asarray(label, dtype='uint8'))
        img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
        img.save(filepath, 'PNG')

    def _dimshuffle(self, input_img, from_axes, to_axes):
        # check axes parameter
        if from_axes.find('0') == -1 or from_axes.find('1') == -1:
            raise ValueError("`from_axes` must contain both axis0 ('0') and"
                            "axis 1 ('1')")
        if to_axes.find('0') == -1 or to_axes.find('1') == -1:
            raise ValueError("`to_axes` must contain both axis0 ('0') and"
                            "axis 1 ('1')")
        if len(from_axes) != len(input_img.shape):
            raise ValueError("Number of axis given by `from_axes` does not match "
                            "the number of axis in `input_img`")

        # handle special cases for channel axis
        to_axes_c = to_axes.find('c')
        from_axes_c = from_axes.find('c')
        # remove channel axis (only grayscale image)
        if to_axes_c == -1 and from_axes_c >= 0:
            if input_img.shape[from_axes_c] != 1:
                raise ValueError('Cannot remove channel axis because size is not '
                                'equal to 1')
            input_img = input_img.squeeze(axis=from_axes_c)
            from_axes = from_axes.replace('c', '')

        # handle special cases for batch axis
        to_axes_b = to_axes.find('b')
        from_axes_b = from_axes.find('b')
        # remove batch axis
        if to_axes_b == -1 and from_axes_b >= 0:
            if input_img.shape[from_axes_b] != 1:
                raise ValueError('Cannot remove batch axis because size is not '
                                'equal to 1')
            input_img = input_img.squeeze(axis=from_axes_b)
            from_axes = from_axes.replace('b', '')

        # add new batch axis (in front)
        if to_axes_b >= 0 and from_axes_b == -1:
            input_img = input_img[np.newaxis]
            from_axes = 'b' + from_axes

        # add new channel axis (in front)
        if to_axes_c >= 0 and from_axes_c == -1:
            input_img = input_img[np.newaxis]
            from_axes = 'c' + from_axes

        return np.transpose(input_img, [from_axes.find(a) for a in to_axes])

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)