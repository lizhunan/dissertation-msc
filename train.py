from src.models.model import EISSegNet
import argparse
import os
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from src.datasets.nyuv2.dataset import NYUv2
from src.datasets.sunrgbd.dataset import SUNRGBD
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
    
    # check dataset
    if args.dataset == 'nyuv2':
        training_dataset = NYUv2(args.dataset_dir, 
                             transform=preprocessing.get_preprocessor(height=args.height, width=args.width, phase='train'), 
                             phase='train')
        val_dataset = NYUv2(args.dataset_dir, 
                            transform=preprocessing.get_preprocessor(height=args.height, width=args.width, phase='test'), 
                            phase='test')
        num_classes = 40
    elif args.dataset == 'sunrgbd':
        training_dataset = SUNRGBD(args.dataset_dir, 
                             transform=preprocessing.get_preprocessor(height=args.height, width=args.width, phase='train'), 
                             phase='train')
        val_dataset = SUNRGBD(args.dataset_dir, 
                            transform=preprocessing.get_preprocessor(height=args.height, width=args.width, phase='test'), 
                            phase='test')
        num_classes = 37
    else:
        NotImplementedError(f'Only nyuv2 and sunrgbd are supported. Got{args.dataset}')
    
    # loading dataset
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # loading model
    model = EISSegNet(dataset=args.dataset, fusion_module=args.fusion_module, 
                      rgb_encoder=args.rgb_encoder , depth_encoder=args.depth_encoder, 
                      upsampling='learned-3x3-zeropad')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    print(model)

    best_model_wts = copy.deepcopy(model.state_dict()) # store best model weighting
    best_miou = 0.
    train_loss_all = [] # store all of train phase loss
    val_loss_all = [] # store all of validation phase loss
    val_miou_all = [] # store all of validation phase miou

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
        best_epoch = 0

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
        val_miou_all.append(val_miou)
        LOG_VAL(epoch + 1, train_loss_all[-1], val_loss_all[-1], val_miou)
        confusion_matrices.reset()

        if ((epoch+1) % 50) == 0:
            print(f'{args.last_ckpt}/ckp_{args.dataset}_{epoch+1}.pth has been saved.')
            torch.save(model.module.state_dict(), 
                       f'{args.last_ckpt}/ckp_{args.fusion_module}_{args.rgb_encoder}_{args.dataset}_{epoch+1}.pth')

        # best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch+1
            best_model_wts = copy.deepcopy(model.module.state_dict())
        
        # durating for each epoch
        time_use = time.time() - epoch_start
        print('Train and val complete in {:.0f}m {:.0f}s'.format(time_use // 60, time_use %60))
        
    train_process = pd.DataFrame(
        data={'epoch':range(1, args.epochs+1),
              'train_loss_all':train_loss_all,
              'val_loss_all':val_loss_all,
              'val_miou':val_miou_all})
    return best_model_wts, best_epoch, train_process

if __name__ == '__main__':
    parser = SegmentationArgumentParser(
        description='Efficient Indoor Scene Segmentation Network (Training).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.set_default_args()
    args = parser.parse_args()

    model, best_epoch, process = train(args)
    
    # save best model
    torch.save(model, f'{args.last_ckpt}/ckp_{args.fusion_module}_{args.rgb_encoder}_{args.dataset}_bst_{best_epoch}.pth')
    print(process)