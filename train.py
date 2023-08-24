from src.models.model import EISSegNet
import argparse
import os
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from src.datasets.nyuv2.dataset import NYUv2
import copy
import time
import pandas as pd
from src.utils import LOG_TRAIN, LOG_VAL
from src.datasets import preprocessing
from src.args import SegmentationArgumentParser
from src.evaluator import CrossEntropyLoss2d, ConfusionMatrix, miou_pytorch

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
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

    # loading model
    model = EISSegNet(upsampling='learned-3x3-zeropad')
    model.to(device)
    print(model)

    best_model_wts = copy.deepcopy(model.state_dict()) # store best model weighting
    best_loss = 1e10 # best loss
    train_loss_all = [] # store all of train phase loss
    val_loss_all = [] # store all of validation phase loss

    # loading weighting
    WEIGHTING_TRAIN_PATH = f'{os.path.dirname(os.path.abspath(__file__))}/src/datasets/{args.dataset}/weighting_train.pickle'
    WEIGHTING_TEST_PATH = f'{os.path.dirname(os.path.abspath(__file__))}/src/datasets/{args.dataset}/weighting_test.pickle'
    if os.path.exists(WEIGHTING_TRAIN_PATH):
        weighting_train = pickle.load(open(WEIGHTING_TRAIN_PATH, 'rb'))
    else:
        raise FileNotFoundError(f'{WEIGHTING_TRAIN_PATH} is not found.')
    if os.path.exists(WEIGHTING_TEST_PATH):
        weighting_test = pickle.load(open(WEIGHTING_TEST_PATH, 'rb'))
    else:
        raise FileNotFoundError(f'{WEIGHTING_TEST_PATH} is not found.')
    
    # loading evaluator
    criterion_train = CrossEntropyLoss2d(weight=weighting_train, device=device)
    criterion_test = CrossEntropyLoss2d(weight=weighting_test, device=device)

    # loading optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        NotImplementedError(f'Only SGD and Adam are supported. Got{args.optimizer}')
    lr_scheduler = OneCycleLR(optimizer, 
                              max_lr=[i['lr'] for i in optimizer.param_groups],
                              total_steps=args.epochs,
                              div_factor=25,
                              pct_start=0.1,
                              anneal_strategy='cos',
                              final_div_factor=1e4)
    # loading confusion matrix
    # set the number of classes according to the dataset
    if args.dataset == 'nyuv2':
        num_classes = 40
    elif args.dataset == 'sunrgbd':
        num_classes = 37
    else:
        raise NotImplementedError(f'Only nyuv2 or sunrgbd are supported for rgb encoder. Got {args.dataset}')
    confusion_matrices = ConfusionMatrix(num_classes)
    miou = miou_pytorch(confusion_matrices)

    # start to train
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        lr_scheduler.step(epoch)
        epoch_start = time.time() # compute time of each epoch
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        val_miou = 0.0

        # train model
        model.train()
        for step, sample in enumerate(train_loader):
            train_start = time.time() # compute time of each batch size
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

            LOG_TRAIN(step+1, epoch+1, train_num, args.batch_size, len(train_loader.dataset), 
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
        data={'epoch':range(args.epochs),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'val_miou':val_miou})
    return best_model_wts, train_process

if __name__ == '__main__':
    parser = SegmentationArgumentParser(
        description='Efficient Indoor Scene Segmentation Network (Training).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.set_default_args()
    args = parser.parse_args()

    model, process = train(args)
    print(process)