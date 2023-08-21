from src.models.model import EISSegNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.datasets.nyuv2.dataset import NYUv2
import numpy as np
import copy
import time
import pandas as pd
from src.utils import LOG_TRAIN, LOG_VAL, Evaluator

epochs = 3
batch = 4

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def train():

    # loading dataset
    training_dataset = NYUv2()
    val_dataset = NYUv2(phase=False)
    train_loader = DataLoader(training_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False)

    # loading model
    model = EISSegNet(num_classes=40, upsampling='learned-3x3-zeropad')

    print('Device:', device)
    model.to(device)
    print(model)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all = []
    val_loss_all = []
    since = time.time()

    LR = 0.0003
    criterion = CrossEntropyLoss2d()
    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-4)
    evaluator = Evaluator(40)
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        train_pa = 0.0
        train_miou = 0.0
        val_pa = 0.0
        val_miou = 0.0

        # train model
        model.train()
        for step, sample in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()
            rgb_img = sample['rgb'].float().to(device)
            depth_img = sample['depth'].float().to(device)
            label = sample['label'].long().to(device)
            out = model(rgb_img, depth_img)

            loss = criterion(out[0], label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
            train_num += len(label)

            out = torch.argmax(out[0], dim=1)
            pred = out.cpu().numpy()
            label = label.cpu().numpy()
            evaluator.add_batch(label, pred)
            train_pa += evaluator.pixel_accuracy()
            train_miou += evaluator.mIou()
            LOG_TRAIN(step+1, epoch+1, train_num, batch, len(train_loader.dataset), loss,time.time() - start, 0)
        train_loss_all.append(train_loss / train_num)
        
        # validate model
        model.eval() 
        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                rgb_img = sample['rgb'].float().to(device)
                depth_img = sample['depth'].float().to(device)
                label = sample['label'].long().to(device)
                out = model(rgb_img, depth_img)

                loss = criterion(out, label)
                val_loss += loss.item() * len(label)
                val_num += len(label)

                out = torch.argmax(out, dim=1)
                pred = out.cpu().numpy()
                label = label.cpu().numpy()
                evaluator.add_batch(label, pred)
                val_pa += evaluator.pixel_accuracy()
                val_miou += evaluator.mIou()
        val_loss_all.append(val_loss / val_num)
        train_miou = train_miou/len(train_loader.dataset)
        train_pa = train_pa/len(train_loader.dataset)
        val_miou = val_miou/len(val_loader.dataset)
        val_pa = val_pa/len(val_loader.dataset)
        LOG_VAL(epoch + 1, 
                train_loss_all[-1], val_loss_all[-1], 
                (train_miou, train_pa),
                (val_miou, val_pa))
                

        # best model
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # durating for each epoch
        time_use = time.time() - since
        print('Train and val complete in {:.0f}m {:.0f}s'.format(time_use // 60, time_use %60))
        
    train_process = pd.DataFrame(
        data={'epoch':range(epochs),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'train_miou':train_miou,
              'train_pa':train_pa,
              'val_miou':val_miou,
              'val_pa':val_pa})
    return best_model_wts, train_process

med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773, 0.972934, 1.481525, 5.342475]

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float().to(device),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        mask = targets_scales > 0
        targets_m = targets_scales.clone()
        targets_m[mask] -= 1
        inputs_scales = inputs_scales.to(device)
        targets_m = targets_m.to(device).long()
        loss_all = self.ce_loss(inputs_scales, targets_m)
        return torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float())

if __name__ == '__main__':
    model, process = train()
    print(process)