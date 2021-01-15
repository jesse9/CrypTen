#!/usr/bin/env python3

"""
To run imagenet example in plaintext mode:

$ python3 -m examples.mpc_imagenet.imagenet --imagenet_folder ~/work/imagenet/ILSVRC2012

"""

import argparse
import logging
import torch
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
