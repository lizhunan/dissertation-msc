import argparse

# if __name__ == '__main__':
#     ### python3 download_data.py  --nyuv2-output-path datasets/nyuv2 --download True
#     from src.datasets.nyuv2.prepare_dataset import Preparation
#     parser = argparse.ArgumentParser(description='Prepare datasets.')
#     parser.add_argument('--nyuv2-output-path', type=str, required=True, help='path where to store dataset')
#     parser.add_argument('--download', type=bool,default=True, help='path where to store dataset')
#     arg = parser.parse_args()

#     preparation = Preparation(arg)
#     preparation()

if __name__ == '__main__':
    ###  python3 download_data.py  --rgbd-output-path datasets/sunrgbd --download True
    from src.datasets.sunrgbd.prepare_dataset import Preparation
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--rgbd-output-path', type=str, required=True, help='path where to store dataset')
    parser.add_argument('--download', type=bool,default=True, help='path where to store dataset')
    arg = parser.parse_args()

    preparation = Preparation(arg)
    preparation()