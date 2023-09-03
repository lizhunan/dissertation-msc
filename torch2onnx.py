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
input = Variable(torch.randn(1, 2, 3, 480, 640)).cuda()
model = EISSegNet(dataset=args.dataset, upsampling='learned-3x3-zeropad')
# load model weights
model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
torch.onnx.export(model, input, 'model_onnx.onnx',opset_version=12, input_names=None, output_names=None, verbose=True)