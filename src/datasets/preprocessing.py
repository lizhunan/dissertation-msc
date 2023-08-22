import numpy as np
import torch
import cv2
import matplotlib
import matplotlib.colors
import torchvision
import torchvision.transforms as transforms

# note that mean and std differ depending on the selected depth_mode
# however, the impact is marginal, therefore, we decided to use the
# stats for refined depth for both cases
# stats for raw: mean: 2769.0187903686697, std: 1350.4174149841133
def get_preprocessor(height=480,
                     width=640,
                     depth_mean=2841.94941272766,
                     depth_std=1417.2594281672277,
                     depth_mode='refined',
                     phase='train',
                     train_random_rescale=(1.0, 1.4)):

    if phase == 'train':
        transform_list = [
            RandomRescale(train_random_rescale),
            RandomCrop(crop_height=height, crop_width=width),
            RandomHSV((0.9, 1.1),
                      (0.9, 1.1),
                      (25, 25)),
            RandomFlip(),
            ToTensor(),
            Normalize(depth_mean=depth_mean,
                      depth_std=depth_std,
                      depth_mode=depth_mode),
            MultiScaleLabel(downsampling_rates=[8, 16, 32])
        ]
    elif phase == 'test':
        transform_list = [
            Rescale(height=height, width=width),
            ToTensor(),
            Normalize(depth_mean=depth_mean,
                      depth_std=depth_std,
                      depth_mode=depth_mode)
        ]
    else:
        raise NotImplementedError(f'Only train and test dataset can be used. Got {phase}')
    
    return transforms.Compose(transform_list)

class Rescale:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']

        rgb = cv2.resize(rgb, (self.width, self.height),
                           interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.width, self.height),
                           interpolation=cv2.INTER_NEAREST)

        sample['rgb'] = rgb
        sample['depth'] = depth

        if 'label' in sample:
            label = sample['label']
            label = cv2.resize(label, (self.width, self.height),
                               interpolation=cv2.INTER_NEAREST)
            sample['label'] = label

        return sample

class ToTensor:
    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']
        rgb = rgb.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype('float32')

        sample['rgb'] = torch.from_numpy(rgb).float()
        sample['depth'] = torch.from_numpy(depth).float()

        if 'label' in sample:
            label = sample['label']
            sample['label'] = torch.from_numpy(label).float()

        return sample
    
class Normalize:
    def __init__(self, depth_mean, depth_std, depth_mode='refined'):
        assert depth_mode in ['refined', 'raw']
        self._depth_mode = depth_mode
        self._depth_mean = [depth_mean]
        self._depth_std = [depth_std]

    def __call__(self, sample):
        rgb, depth = sample['rgb'], sample['depth']
        rgb = rgb / 255
        rgb = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(rgb)
        if self._depth_mode == 'raw':
            depth_0 = depth == 0

            depth = torchvision.transforms.Normalize(
                mean=self._depth_mean, std=self._depth_std)(depth)

            depth[depth_0] = 0

        else:
            depth = torchvision.transforms.Normalize(
                mean=self._depth_mean, std=self._depth_std)(depth)

        sample['rgb'] = rgb
        sample['depth'] = depth

        return sample
    
class RandomRescale:
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        rgb, depth, label = sample['rgb'], sample['depth'], sample['label']

        target_scale = np.random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))

        rgb = cv2.resize(rgb, (target_width, target_height),
                           interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (target_width, target_height),
                           interpolation=cv2.INTER_NEAREST)

        sample['rgb'] = rgb
        sample['depth'] = depth
        sample['label'] = label

        return sample


class RandomCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.rescale = Rescale(self.crop_height, self.crop_width)

    def __call__(self, sample):
        rgb, depth, label = sample['rgb'], sample['depth'], sample['label']
        h = rgb.shape[0]
        w = rgb.shape[1]
        if h <= self.crop_height or w <= self.crop_width:
            # simply rescale instead of random crop as image is not large enough
            sample = self.rescale(sample)
        else:
            i = np.random.randint(0, h - self.crop_height)
            j = np.random.randint(0, w - self.crop_width)
            rgb = rgb[i:i + self.crop_height, j:j + self.crop_width, :]
            depth = depth[i:i + self.crop_height, j:j + self.crop_width]
            label = label[i:i + self.crop_height, j:j + self.crop_width]
            sample['rgb'] = rgb
            sample['depth'] = depth
            sample['label'] = label
        return sample


class RandomHSV:
    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['rgb']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h = img_hsv[:, :, 0]
        img_s = img_hsv[:, :, 1]
        img_v = img_hsv[:, :, 2]

        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        sample['rgb'] = img_new

        return sample


class RandomFlip:
    def __call__(self, sample):
        rgb, depth, label = sample['rgb'], sample['depth'], sample['label']
        if np.random.rand() > 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        sample['rgb'] = rgb
        sample['depth'] = depth
        sample['label'] = label

        return sample

class MultiScaleLabel:
    def __init__(self, downsampling_rates=None):
        if downsampling_rates is None:
            self.downsampling_rates = [8, 16, 32]
        else:
            self.downsampling_rates = downsampling_rates

    def __call__(self, sample):
        label = sample['label']

        h, w = label.shape

        sample['label_down'] = dict()

        # Nearest neighbor interpolation
        for rate in self.downsampling_rates:
            label_down = cv2.resize(label.numpy(), (w // rate, h // rate),
                                    interpolation=cv2.INTER_NEAREST)
            sample['label_down'][rate] = torch.from_numpy(label_down)

        return sample
