import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from src.args import SegmentationArgumentParser
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1

        # Load the Onnx model and parse it to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        return builder.build_cuda_engine(network)

def inference_with_trt(onnx_path, rgb_tensor, depth_tensor):
    # Build the TensorRT engine
    engine = build_engine(onnx_path)
    if not engine:
        return

    # Allocate buffers for input and output
    h_input_rgb = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_input_depth = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(trt.float32))
    
    d_input_rgb = cuda.mem_alloc(h_input_rgb.nbytes)
    d_input_depth = cuda.mem_alloc(h_input_depth.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Transfer input data to the GPU
    cuda.memcpy_htod_async(d_input_rgb, rgb_tensor, stream)
    cuda.memcpy_htod_async(d_input_depth, depth_tensor, stream)

    # Run inference
    with engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_input_rgb), int(d_input_depth), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

    stream.synchronize()

    return h_output


if __name__ == '__main__':
    parser = SegmentationArgumentParser(
        description='Efficient Indoor Scene Segmentation Network (Inference with TensorRT).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.set_default_args()
    args = parser.parse_args()

    # Assuming RGB and Depth tensors are preprocessed
    rgb_tensor = None  # Replace with preprocessed tensor
    depth_tensor = None  # Replace with preprocessed tensor
    output = inference_with_trt(args.onnx_path, rgb_tensor, depth_tensor)

    # Handle the output as required
    # ...
