from src.models.model import EISSegNet
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from src.datasets.nyuv2.dataset import NYUv2
import numpy as np
import copy
import time
import pandas as pd
from src.utils import LOG_TRAIN, LOG_VAL, compute_class_weights, ConfusionMatrixPytorch, miou_pytorch
from src.datasets import preprocessing
from src.args import SegmentationArgumentParser

epochs = 300
batch = 2

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def train(args):

    # loading dataset
    training_dataset = NYUv2(args.dataset_dir, 
                             transform=preprocessing.get_preprocessor(height=args.height, width=args.width, phase='train'), 
                             phase='train')
    val_dataset = NYUv2(args.dataset_dir, 
                        transform=preprocessing.get_preprocessor(height=args.height, width=args.width, phase='test'), 
                        phase='test')
    train_loader = DataLoader(training_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    # loading model
    model = EISSegNet(upsampling='learned-3x3-zeropad')

    print('Device:', device)
    model.to(device)
    print(model)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all = []
    val_loss_all = []

    LR = 0.001
    weighting_train = compute_class_weights(path='/cs/home/psxzl18/dissertation-msc/src/datasets/nyuv2/weighting_median_frequency_1+40_train.pickle')
    weighting_test = compute_class_weights(path='/cs/home/psxzl18/dissertation-msc/src/datasets/nyuv2/weighting_linear_1+40_test.pickle')
    criterion_train = CrossEntropyLoss2d(weight=weighting_train)
    criterion_test = CrossEntropyLoss2d(weight=weighting_test)
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=LR, weight_decay=0.0001, nesterov=True)
    lr_scheduler = OneCycleLR(optimizer, 
                              max_lr=[i['lr'] for i in optimizer.param_groups],
                              total_steps=epochs,
                              div_factor=25,
                              pct_start=0.1,
                              anneal_strategy='cos',
                              final_div_factor=1e4)
    confusion_matrices = ConfusionMatrixPytorch(40)
    miou = miou_pytorch(confusion_matrices)
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        lr_scheduler.step(epoch)
        epoch_start = time.time()
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        val_miou = 0.0

        # train model
        model.train()
        for step, sample in enumerate(train_loader):
            train_start = time.time()
            optimizer.zero_grad()
            rgb_img = sample['rgb'].float().to(device)
            depth_img = sample['depth'].float().to(device)
            label = sample['label'].long().to(device)
            out = model(rgb_img, depth_img)

            loss = criterion_train(out[0], label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
            train_num += len(label)

            LOG_TRAIN(step+1, epoch+1, train_num, batch, len(train_loader.dataset), 
                      loss,time.time() - train_start, lr_scheduler.get_lr())
        train_loss_all.append(train_loss / train_num)
        
        # validate model
        model.eval() 
        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                rgb_img = sample['rgb'].float().to(device)
                depth_img = sample['depth'].float().to(device)
                label = sample['label'].long().to(device)
                out = model(rgb_img, depth_img)

                loss = criterion_test(out, label)
                val_loss += loss.item() * len(label)
                val_num += len(label)

                out = torch.argmax(out, dim=1)
                mask = label > 0
                label = torch.masked_select(label, mask)
                out = torch.masked_select(out, mask.to(device))
                label -= 1
                out = out.cpu().numpy()
                label = label.cpu().numpy()
                confusion_matrices.update(torch.from_numpy(label), torch.from_numpy(out))

        val_loss_all.append(val_loss / val_num)
        val_miou = miou.compute().data.numpy()
        LOG_VAL(epoch + 1, train_loss_all[-1], val_loss_all[-1], val_miou)
        confusion_matrices.reset()

        # best model
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # durating for each epoch
        time_use = time.time() - epoch_start
        print('Train and val complete in {:.0f}m {:.0f}s'.format(time_use // 60, time_use %60))
        
    train_process = pd.DataFrame(
        data={'epoch':range(epochs),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'val_miou':val_miou})
    return best_model_wts, train_process

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           reduction='none', ignore_index=-1)
        self.ce_loss.to(device)

    def forward(self, inputs_scales, targets_scales):
        targets_m = targets_scales.clone()
        targets_m -= 1
        inputs_scales = inputs_scales.to(device)
        targets_m = targets_m.to(device).long()
        loss_all = self.ce_loss(inputs_scales, targets_m)

        number_of_pixels_per_class = \
                torch.bincount(targets_scales.flatten().type(self.dtype),
                               minlength=self.num_classes)
        divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void

        return torch.sum(loss_all) / divisor_weighted_pixel_sum

if __name__ == '__main__':
    parser = SegmentationArgumentParser(
        description='Efficient Indoor Scene Segmentation Network (Training).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.set_default_args()
    args = parser.parse_args()

    model, process = train(args)
    print(process)