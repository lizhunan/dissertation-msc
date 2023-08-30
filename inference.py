from src.args import SegmentationArgumentParser
import argparse
from src.models.model import EISSegNet
import skimage.transform
import torch
import os
import numpy as np
import imageio
import torchvision
import matplotlib.pyplot as plt
import time
from src.evaluator import ConfusionMatrix, miou_pytorch

NYUV2_LABELS = [(0, 0, 0),
                 (148, 65, 137), (255, 116, 69), (86, 156, 137),
                 (202, 179, 158), (155, 99, 235), (161, 107, 108),
                 (133, 160, 103), (76, 152, 126), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]

SUNRGBD_LABELS = [(0, 0, 0), (119, 119, 119), (244, 243, 131),
                    (137, 28, 157), (150, 255, 255), (54, 114, 113),
                    (0, 0, 176), (255, 69, 0), (87, 112, 255), (0, 163, 33),
                    (255, 150, 255), (255, 180, 10), (101, 70, 86),
                    (38, 230, 0), (255, 120, 70), (117, 41, 121),
                    (150, 255, 0), (132, 0, 255), (24, 209, 255),
                    (191, 130, 35), (219, 200, 109), (154, 62, 86),
                    (255, 190, 190), (255, 0, 255), (152, 163, 55),
                    (192, 79, 212), (230, 230, 230), (53, 130, 64),
                    (155, 249, 152), (87, 64, 34), (214, 209, 175),
                    (170, 0, 59), (255, 0, 0), (193, 195, 234), (70, 72, 115),
                    (255, 255, 0), (52, 57, 131), (12, 83, 45)]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def inference(args):
    
    # loading model and checkpoint
    model = EISSegNet(dataset=args.dataset, upsampling='learned-3x3-zeropad')
    model.load_state_dict(torch.load(args.ckpt_path))
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)
    if args.dataset == 'nyuv2':
        img_dir_test, depth_dir_test, label_dir_test = _load_nyuv2(args.dataset_dir)
        num_classes = 40
    elif args.dataset == 'sunrgbd':
        img_dir_test, depth_dir_test, label_dir_test = _load_sunrgbd(args.dataset_dir)
        num_classes = 37
    else:
        raise NotImplementedError(f'Only nyuv2 or sunrgbd are supported for rgb encoder. Got {args.dataset}')
    img_dir_test = img_dir_test[:args.num_samples]
    depth_dir_test = depth_dir_test[:args.num_samples]
    label_dir_test = label_dir_test[:args.num_samples]

    # loading confusion matrix
    confusion_matrices = ConfusionMatrix(num_classes)
    miou = miou_pytorch(confusion_matrices)

    with torch.no_grad():
        img_dir_test 
        mark = round(time.time())
        for i, (rgb_path, depth_path, label_path) in enumerate(zip(img_dir_test, depth_dir_test, label_dir_test)):
            rgb_img = imageio.imread(rgb_path)
            depth_img = imageio.imread(depth_path)
            if args.dataset == 'nyuv2':
                label_orig = imageio.imread(label_path)
            elif args.dataset == 'sunrgbd':
                label_orig = np.load(label_path)
                # if dataset is sunrgbd, resize image to (480, 640)
                # Bi-linear
                rgb_img = skimage.transform.resize(rgb_img, (480, 640), order=1,
                                                mode='reflect', preserve_range=True)
                # Nearest-neighbor
                depth_img = skimage.transform.resize(depth_img, (480, 640), order=0,
                                                mode='reflect', preserve_range=True)
                label_orig = skimage.transform.resize(label_orig, (480, 640), order=0,
                                                mode='reflect', preserve_range=True)
            else:
                raise NotImplementedError(f'Only nyuv2 or sunrgbd are supported for rgb encoder. Got {args.dataset}')

            # normalization
            rgb = rgb_img / 255
            rgb = torch.from_numpy(rgb).float()
            depth = torch.from_numpy(depth_img).float()
            rgb = rgb.permute(2, 0, 1)
            depth.unsqueeze_(0)
            rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])(rgb)
            depth = torchvision.transforms.Normalize(mean=[2841.94941272766],
                                                std=[1417.2594281672277])(depth)
            
            # to tensor
            rgb = rgb.to(device).unsqueeze_(0)
            depth = depth.to(device).unsqueeze_(0)
            label_miou = torch.from_numpy(label_orig).float().to(device).unsqueeze_(0)

            start = time.time()
            pred = model(rgb, depth)
            end = time.time() - start 
            if args.dataset == 'nyuv2':
                output = _color_label(torch.max(pred, 1)[1] + 1, NYUV2_LABELS)
            elif args.dataset == 'sunrgbd':
                output = _color_label(torch.max(pred, 1)[1] + 1, SUNRGBD_LABELS)
            else:
                raise NotImplementedError(f'Only nyuv2 or sunrgbd are supported for rgb encoder. Got {args.dataset}')

            # computing miou
            pred_miou = torch.argmax(pred, dim=1)
            mask = label_miou > 0
            label_miou = torch.masked_select(label_miou.long(), mask)
            pred_miou = torch.masked_select(pred_miou, mask.to(device))
            label_miou -= 1
            pred_miou = pred_miou.cpu().numpy()
            label_miou = label_miou.cpu().numpy()
            confusion_matrices.update(torch.from_numpy(label_miou), torch.from_numpy(pred_miou))
            val_miou = miou.compute().data.numpy()
            confusion_matrices.reset()

            # save result to args.results_dir
            fig, axs = plt.subplots(1, 4, figsize=( 16, 4))
            [ax.set_axis_off() for ax in axs.ravel()]
            axs[0].imshow(rgb_img)
            axs[1].imshow(depth_img, cmap='gray')
            axs[2].imshow(label_orig, cmap='gray')
            axs[3].imshow(output)
            axs[0].set_title('RGB Image')
            axs[1].set_title('Depth Image')
            axs[2].set_title('Label Original')
            axs[3].set_title('Prediction')
            fig.suptitle(f'Result on {args.dataset}(mIoU: {val_miou:0.4f}, Duration: {end:0.4f}s, FPS: {1/end:0.4f})', fontsize=16)
            plt.savefig(os.path.join(args.results_dir, f'result_{args.dataset}_{mark}_{i+1}.png'), dpi=150)

def _color_label(label, mask):
    h, w = label.shape[1:3]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            colored[i, j] = mask[label[0, i, j]]
    return colored

def _load_nyuv2(data_dir):
    try:
        img_dir_test = []
        depth_dir_test = []
        label_dir_test = []
        tmp_file = os.path.join(data_dir, 'test.txt')
        with open(tmp_file, 'r') as f:
            test_imgs = f.read().splitlines()
            for img in test_imgs:
                img_dir_test.append(f'{data_dir}/test/rgb/{img}.png')
                depth_dir_test.append(f'{data_dir}/test/depth/{img}.png')
                label_dir_test.append(f'{data_dir}/test/labels_40/{img}.png')
        return img_dir_test, depth_dir_test, label_dir_test
    except:
        raise IOError(f'{tmp_file} not found.')
    
def _load_sunrgbd(data_dir):
    try:
        img_dir_test = []
        depth_dir_test = []
        label_dir_test = []
        tmp_file = os.path.join(data_dir, 'test_rgb.txt')
        with open(tmp_file, 'r') as f:
            test_imgs = f.read().splitlines()
            for img in test_imgs:
                img_dir_test.append(f'{data_dir}/{img}')
        tmp_file = os.path.join(data_dir, 'test_depth.txt')
        with open(tmp_file, 'r') as f:
            test_imgs = f.read().splitlines()
            for img in test_imgs:
                depth_dir_test.append(f'{data_dir}/{img}')
        tmp_file = os.path.join(data_dir, 'test_label.txt')
        with open(tmp_file, 'r') as f:
            test_imgs = f.read().splitlines()
            for img in test_imgs:
                label_dir_test.append(f'{data_dir}/{img}')
        return img_dir_test, depth_dir_test, label_dir_test
    except:
        raise IOError(f'{tmp_file} not found.')

if __name__ == '__main__':
    parser = SegmentationArgumentParser(
        description='Efficient Indoor Scene Segmentation Network (Inference).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.set_default_args()
    args = parser.parse_args()

    inference(args)