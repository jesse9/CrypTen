#!/usr/bin/env python3

"""
To run imagenet example in plaintext mode:

$ python3 -m examples.mpc_imagenet.imagenet --imagenet_folder ~/work/imagenet/ILSVRC2012

"""

import argparse
import io
import logging
import onnx
import torch
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from examples.meters import AccuracyMeter


# input arguments:
parser = argparse.ArgumentParser(description="Plaintext inference of vision models")
parser.add_argument(
    "--model",
    default="resnet18",
    type=str,
    help="torchvision model to use for inference (default: resnet18)",
)
parser.add_argument(
    "--imagenet_folder",
    default=None,
    type=str,
    help="folder containing the ImageNet dataset",
)
parser.add_argument(
    "--num_samples",
    default=None,
    type=int,
    help="number of samples to test on (default: all)",
)


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


@sym_help.parse_args("v", "i", "none")
def _onnx_crypten_softmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's Softmax module to a Softmax module in
    the ONNX model. It overrides PyTorch's default conversion of Softmax module
    to a sequence of Exp, ReduceSum and Div modules, since this default
    conversion can cause numerical overflow when applied to CrypTensors.
    """
    result = g.op("Softmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "i", "none")
def _onnx_crypten_logsoftmax(g, input, dim, dtype=None):
    """
    This function converts PyTorch's LogSoftmax module to a LogSoftmax module in
    the ONNX model. It overrides PyTorch's default conversion of LogSoftmax module
    to avoid potentially creating Transpose operators.
    """
    result = g.op("LogSoftmax", input, axis_i=dim)
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = sym_help._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return result


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_dropout(g, input, p, train):
    """
    This function converts PyTorch's Dropout module to a Dropout module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the Dropout module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore Dropout modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the Dropout module needs to be included in the
    CrypTen-specific conversion.
    """
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


@sym_help.parse_args("v", "f", "i")
def _onnx_crypten_feature_dropout(g, input, p, train):
    """
    This function converts PyTorch's DropoutNd module to a DropoutNd module in the ONNX
    model. It overrides PyTorch's default implementation to ignore the DropoutNd module
    during the conversion. PyTorch assumes that ONNX models are only used for
    inference and therefore DropoutNd modules are not required in the ONNX model.
    However, CrypTen needs to convert ONNX models to trainable
    CrypTen models, and so the DropoutNd module needs to be included in the
    CrypTen-specific conversion.
    """
    r, _ = g.op("DropoutNd", input, ratio_f=p, outputs=2)
    return r


def _update_onnx_symbolic_registry():
    """
    Updates the ONNX symbolic registry for operators that need a CrypTen-specific
    implementation and custom operators.
    """
    for version_key, version_val in sym_registry._registry.items():
        for function_key in version_val.keys():
            if function_key == "softmax":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_softmax
            if function_key == "log_softmax":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_logsoftmax
            if function_key == "dropout":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_dropout
            if function_key == "feature_dropout":
                sym_registry._registry[version_key][
                    function_key
                ] = _onnx_crypten_feature_dropout


def _export_pytorch_model(f, pytorch_model, dummy_input):
    """Returns a Binary I/O stream containing exported model"""
    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "enable_onnx_checker": False,
        "input_names": ["input"],
        "output_names": ["output"],
    }
    try:
        # current version of PyTorch requires us to use `enable_onnx_checker`
        torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    except TypeError:
        # older versions of PyTorch require us to NOT use `enable_onnx_checker`
        kwargs.pop("enable_onnx_checker")
        torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    return f


def _from_pytorch_to_bytes(pytorch_model, dummy_input):
    """Returns I/O stream containing onnx graph with crypten specific ops"""
    # TODO: Currently export twice because the torch-to-ONNX symbolic registry
    # only gets created on the first call.
    with io.BytesIO() as f:
        _export_pytorch_model(f, pytorch_model, dummy_input)

    # update ONNX symbolic registry with CrypTen-specific functions
    _update_onnx_symbolic_registry()

    # export again so the graph is created with CrypTen-specific registry
    f = io.BytesIO()
    f = _export_pytorch_model(f, pytorch_model, dummy_input)
    f.seek(0)
    return f


def run_experiment(
    model_name,
    imagenet_folder=None,
    num_samples=None,
):
    """Runs inference using specified vision model on specified dataset."""
    model = getattr(models, model_name)(pretrained=True)
    model.eval()
    dataset = datasets.ImageNet(imagenet_folder, split="val")

    # define appropriate transforms:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    to_tensor_transform = transforms.ToTensor()
    dummy_input = to_tensor_transform(dataset[0][0])
    dummy_input.unsqueeze_(0)
    onnx_bytes_io = _from_pytorch_to_bytes(model, dummy_input)
    onnx_bytes_io.seek(0)
    onnx_model_proto = onnx.load(onnx_bytes_io)
    onnx.checker.check_model(onnx_model_proto)
    print(onnx.helper.printable_graph(onnx_model_proto.graph))

    # loop over dataset:
    meter = AccuracyMeter()
    for idx, sample in enumerate(dataset):
        # preprocess sample:
        image, target = sample
        image = transform(image)
        image.unsqueeze_(0)
        target = torch.LongTensor([target])

        output = model(image)
        meter.add(output, target)

        # progress:
        logging.info(
            "[sample %d of %d] Accuracy: %f" % (idx + 1, len(dataset), meter.value()[1])
        )
        if num_samples is not None and idx == num_samples - 1:
            break

    # print final accuracy:
    logging.info("Accuracy on all %d samples: %f" % (len(dataset), meter.value()[1]))


def main():
    args = parser.parse_args()
    run_experiment(
        args.model,
        imagenet_folder=args.imagenet_folder,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
