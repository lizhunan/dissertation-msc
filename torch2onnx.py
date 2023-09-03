import torch
import argparse
from src.models.model import EISSegNet
from src.args import SegmentationArgumentParser
from torch.autograd import Variable

parser = SegmentationArgumentParser(
        description='Efficient Indoor Scene Segmentation Network (troch2onnx).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.set_default_args()
args = parser.parse_args()

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch  -->  onnx
rgb_dummy_input = torch.randn(1, 3, 480, 640).cuda()
depth_dummy_input = torch.randn(1, 1, 480, 640).cuda()
model = EISSegNet(upsampling='learned-3x3-zeropad').cuda()
# load model weights
model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
torch.onnx.export(model, (rgb_dummy_input, depth_dummy_input), 'eissegnet.onnx')