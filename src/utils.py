import os
import pickle
import numpy as np
import torch
import numbers
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, MetricsLambda

def LOG_TRAIN(index, epoch, train_num, batch, dataset_size, loss, time_, learning_rates):
    log = 'Train Epoch: {:>3}  Step: {:>3}  [{:>4}/{:>4} ({: 5.1f}%)]  Loss: {:0.6f}'.format(
        epoch, index, train_num, dataset_size, 100. * train_num/dataset_size, loss
    )
    log += '  Learning Rate: {:>6}'.format(round(learning_rates, 10))
    log += '  [{:0.3f}s every {:>4} data]'.format(time_, batch)

    print(log, flush=True)

def LOG_VAL(epoch, train_loss_all, val_loss_all, val_eval):
    log = 'Epoch {:3}  Train Loss: {:.4f}\n'.format(epoch, train_loss_all)
    log += 'Epoch {:3}  Validation Loss: {:.4f}\n'.format(epoch, val_loss_all)
    log += 'mIoU Test: {:0.20f} ({: 5.1f}%)\n'.format(val_eval, 100. * val_eval)

    print(log, flush=True)

class Evaluator:

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def pixel_accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    
    def mIou(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class ConfusionMatrixPytorch(Metric):
    def __init__(self,
                 num_classes,
                 average=None,
                 output_transform=lambda x: x):
        if average is not None and average not in ("samples", "recall",
                                                   "precision"):
            raise ValueError("Argument average can None or one of "
                             "['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        if self.num_classes < np.sqrt(2**8):
            self.dtype = torch.uint8
        elif self.num_classes < np.sqrt(2**16 / 2):
            self.dtype = torch.int16
        elif self.num_classes < np.sqrt(2**32 / 2):
            self.dtype = torch.int32
        else:
            self.dtype = torch.int64
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrixPytorch, self).__init__(
            output_transform=output_transform
        )

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes,
                                            self.num_classes,
                                            dtype=torch.int64,
                                            device='cpu')
        self._num_examples = 0

    def update(self, y, y_pred, num_examples=1):
        assert len(y) == len(y_pred), ('label and prediction need to have the'
                                       ' same size')
        self._num_examples += num_examples

        y = y.type(self.dtype)
        y_pred = y_pred.type(self.dtype)

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices,
                           minlength=self.num_classes ** 2)
        m = m.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one '
                                     'example before it can be computed.')
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix


def iou_pytorch(cm, ignore_index=None):
    if not isinstance(cm, ConfusionMatrixPytorch):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, "
                        "but given {}".format(type(cm)))

    if ignore_index is not None:
        if (not (isinstance(ignore_index, numbers.Integral)
                 and 0 <= ignore_index < cm.num_classes)):
            raise ValueError("ignore_index should be non-negative integer, "
                             "but given {}".format(ignore_index))

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError("ignore_index {} is larger than the length "
                                 "of IoU vector {}"
                                 .format(ignore_index, len(iou_vector)))
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou


def miou_pytorch(cm, ignore_index=None):
    return iou_pytorch(cm=cm, ignore_index=ignore_index).mean()

def compute_class_weights(path, classes=40, phase='train', weight_mode='median_frequency'):

    # weighting_path = os.path.join(
    #     self.source_path, f'weighting_{weight_mode}_'
    #                       f'1+{classes}')
    # weighting_path += f'_{phase}.pickle'

    weighting_path = path

    if os.path.exists(weighting_path):
        weighting = pickle.load(open(weighting_path, 'rb'))
        print(f'Using {weighting_path} as class weighting')
        return weighting
    else:
        raise FileNotFoundError(f'Weighting file not found. Got{weighting_path}')