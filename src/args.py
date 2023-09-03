import argparse


class SegmentationArgumentParser(argparse.ArgumentParser):

    def set_default_args(self):
        
        # dataset
        self.add_argument('--dataset', default='nyuv2',
                          choices=['sunrgbd',
                                   'nyuv2'])
        self.add_argument('--dataset_dir',
                          default=None,
                          help='Path to dataset root.',)
        self.add_argument('--height', type=int, default=480,
                          help='height of the training images. '
                               'Images will be resized to this height.')
        self.add_argument('--width', type=int, default=640,
                          help='width of the training images. '
                               'Images will be resized to this width.')

        # model
        self.add_argument('--rgb_encoder', default='convnext_b',
                          choices=['convnext_b',
                                   'convnext_t'])
        self.add_argument('--depth_encoder', default='convnext_b',
                          choices=['convnext_b',
                                   'convnext_t'])
        self.add_argument('--fusion_module', default='ECA',
                          choices=['ECA',
                                   'SE'])

        # train
        self.add_argument('--last_ckpt', default='./trained_models', type=str, metavar='PATH',
                          help='path to latest checkpoint')
        self.add_argument('--batch_size', type=int, default=4,
                          help='batch size for training')
        self.add_argument('--epochs', default=500, type=int, metavar='N',
                          help='number of total epochs to run')
        self.add_argument('--lr', '--learning-rate', default=0.001,
                          type=float,
                          help='maximum learning rate. When using one_cycle '
                               'as --lr_scheduler lr will first increase to '
                               'the value provided and then slowly decrease.')
        self.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                          help='weight decay')
        self.add_argument('--momentum', default=0.9, type=float, metavar='M',
                          help='momentum')
        self.add_argument('--optimizer', type=str, default='SGD',
                          choices=['SGD', 'Adam'])
        
        # inference
        self.add_argument('--results_dir',
                          default='./results')
        self.add_argument('--ckpt_path',
                          default=None)
        self.add_argument('--num_samples', type=int,
                          default=1,
                          help='the amount of inferences.')
        
        # others
        