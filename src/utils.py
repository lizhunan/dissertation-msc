
import numpy as np

def LOG_TRAIN(index, epoch, train_num, batch, dataset_size, loss, time_, learning_rates):
    log = 'Train Epoch: {:>3}  Step: {:>3}  [{:>4}/{:>4} ({: 5.1f}%)]  Loss: {:0.6f}'.format(
        epoch, index, train_num, dataset_size, 100. * train_num/dataset_size, loss
    )
    log += '  [{:0.3f}s every {:>4} data]'.format(time_, batch)

    print(log, flush=True)

def LOG_VAL(epoch, train_loss_all, val_loss_all, train_eval, val_eval):
    log = 'Epoch {:3}  Train Loss: {:.4f}\n'.format(epoch, train_loss_all)
    log += 'Epoch {:3}  Validation Loss: {:.4f}\n'.format(epoch, val_loss_all)
    log += 'mIoU Train: {:0.20f} ({: 5.1f}%)\n'.format(train_eval[0], 100. * train_eval[0])
    log += 'Pixel Accuracy Train: {:0.20f} ({: 5.1f}%)\n'.format(train_eval[1], 100. * train_eval[1])
    log += 'mIoU Test: {:0.20f} ({: 5.1f}%)\n'.format(val_eval[0], 100. * val_eval[0])
    log += 'Pixel Accuracy Test: {:0.20f} ({: 5.1f}%)'.format(val_eval[1], 100. * val_eval[1])

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