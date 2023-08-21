import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose


if __name__ == '__main__':
    from src.datasets.nyuv2.dataset import NYUv2

    dataset = NYUv2()
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True,
                              num_workers=0, pin_memory=False)
    
    for step, sample in enumerate(train_loader):
        if step > 0:
            break

    print("rgb.shape:",sample['rgb'].shape)
    print("depth.shape:",sample['depth'].shape)
    print("label.shape:",sample['label'].shape)

    # b_rbg_numpy = sample['rgb'].data.numpy()
    # b_depth_numpy = sample['depth'].data.numpy()

    # import matplotlib.pyplot as plt
    # b_y_numpy = sample['label'].data.numpy()
    # plt.imsave('tmp/brbg-train.png', b_rbg_numpy[1])
    # plt.imsave('tmp/bdepth-train.png', b_depth_numpy[1])
    # plt.imsave('tmp/by-train.png', b_y_numpy[1])


# if __name__ == '__main__':
#     from src.datasets.sunrgbd.dataset import SUNRGBD

#     transform = transforms.Compose([transforms.Resize((640, 480)), transforms.ToTensor()])
#     dataset = SUNRGBD(transform=transform)
#     train_loader = DataLoader(dataset, batch_size=8, shuffle=True,
#                               num_workers=0, pin_memory=False)
    
#     for step, sample in enumerate(train_loader):
#         if step > 0:
#             break

#     print("rgb.shape:",sample['rgb'].shape)
#     print("depth.shape:",sample['depth'].shape)
#     print("label.shape:",sample['label'].shape)

#     b_rbg_numpy = sample['rgb'].data.numpy()
#     b_depth_numpy = sample['depth'].data.numpy()

#     import matplotlib.pyplot as plt
#     b_y_numpy = sample['label'].data.numpy()
#     plt.imsave('tmp/srbg-train.png', b_rbg_numpy[1])
#     plt.imsave('tmp/sdepth-train.png', b_depth_numpy[1])
#     plt.imsave('tmp/sy-train.png', b_y_numpy[1])
