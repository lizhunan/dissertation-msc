"""
This code is partially adapted from RedNet
    (https://github.com/JinDongJiang/RedNet)
"""

import torch
import torch.nn as nn
import numpy as np
import numbers
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, MetricsLambda

class CrossEntropyLoss2d(nn.Module):
    """CrossEntropyLoss2d is based on nn.CrossEntropyLoss with weight.
    as for relatively large num_classes(>2**8), using unit16, 
    other cases using unit8.
    """
    def __init__(self, weight, device):
        super(CrossEntropyLoss2d, self).__init__()
        self.device = device
        self.weight = torch.tensor(weight).to(device)
        # +1 for void
        self.num_classes = len(self.weight) + 1
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
        inputs_scales = inputs_scales.to(self.device)
        targets_m = targets_m.to(self.device).long()
        loss_all = self.ce_loss(inputs_scales, targets_m)

        number_of_pixels_per_class = \
                torch.bincount(targets_scales.flatten().type(self.dtype),
                               minlength=self.num_classes)
        # without void
        divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[1:] * self.weight)  

        return torch.sum(loss_all) / divisor_weighted_pixel_sum

class ConfusionMatrix(Metric):
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
        super(ConfusionMatrix, self).__init__(
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
    if not isinstance(cm, ConfusionMatrix):
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
