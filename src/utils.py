import numpy as np

def LOG_TRAIN(index, epoch, train_num, batch, dataset_size, loss, time_, learning_rates):
    log = 'Train Epoch: {:>3}  Step: {:>3}  [{:>4}/{:>4} ({: 5.1f}%)]  Loss: {:0.6f}'.format(
        epoch, index, train_num, dataset_size, 100. * train_num/dataset_size, loss
    )
    for i, lr in enumerate(learning_rates):
        log += '  lr_{}: {:>6}'.format(i, round(lr, 10))
    log += '  [{:0.3f}s every {:>4} data]'.format(time_, batch)

    print(log, flush=True)

def LOG_VAL(epoch, train_loss_all, val_loss_all, val_eval):
    log = 'Epoch {:3}  Train Loss: {:.4f}\n'.format(epoch, train_loss_all)
    log += 'Epoch {:3}  Validation Loss: {:.4f}\n'.format(epoch, val_loss_all)
    log += 'mIoU Test: {:0.20f} ({: 5.1f}%)\n'.format(val_eval, 100. * val_eval)

    print(log, flush=True)