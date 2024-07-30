#!/usr/bin/env python3
""" ImageNet Validation Script in TensorFlow

This is a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets using TensorFlow.
It prioritizes canonical TensorFlow, standard Python style, and good performance.

Hacked together by OpenAI's ChatGPT based on Ross Wightman's PyTorch script
"""

import argparse
import json
import logging
import os
import time
from collections import OrderedDict

import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50, ResNet101, ResNet152, VGG16, VGG19, InceptionV3, Xception
from keras.preprocessing import image_dataset_from_directory
from keras.applications.resnet import preprocess_input, decode_predictions

_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='TensorFlow ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None, help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)')
parser.add_argument('--model', '-m', metavar='NAME', default='ResNet50', help='model architecture (default: ResNet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int, metavar='N', help='Input image dimension (default: 224)')
parser.add_argument('--device', default='GPU', type=str, help="Device to use (default: GPU)")
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME', help='Output file for validation results (JSON)')

args = parser.parse_args()

def get_model(name):
    if name == 'ResNet50':
        return ResNet50(weights='imagenet')
    elif name == 'ResNet101':
        return ResNet101(weights='imagenet')
    elif name == 'ResNet152':
        return ResNet152(weights='imagenet')
    elif name == 'VGG16':
        return VGG16(weights='imagenet')
    elif name == 'VGG19':
        return VGG19(weights='imagenet')
    elif name == 'InceptionV3':
        return InceptionV3(weights='imagenet')
    elif name == 'Xception':
        return Xception(weights='imagenet')
    else:
        raise ValueError("Unsupported model architecture")

def main():
    setup_default_logging()
    
    if args.device == 'GPU':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            except RuntimeError as e:
                _logger.error(e)

    data_dir = args.data or args.data_dir
    if not data_dir:
        raise ValueError("Data directory must be specified with --data or --data-dir")

    dataset = image_dataset_from_directory(
        data_dir,
        shuffle=False,
        batch_size=args.batch_size,
        image_size=(args.img_size, args.img_size)
    )

    model = get_model(args.model)
    _logger.info(f'Model {args.model} loaded')

    batch_time = []
    losses = []
    top1 = []
    top5 = []

    for batch_idx, (images, labels) in enumerate(dataset):
        start_time = time.time()
        images = preprocess_input(images)
        preds = model.predict(images)
        end_time = time.time()

        batch_time.append(end_time - start_time)
        
        top1_acc = tf.keras.metrics.top_k_categorical_accuracy(labels, preds, k=1)
        top5_acc = tf.keras.metrics.top_k_categorical_accuracy(labels, preds, k=5)
        
        top1.append(top1_acc)
        top5.append(top5_acc)

        if batch_idx % 10 == 0:
            _logger.info(f'Batch [{batch_idx}/{len(dataset)}]: Time {batch_time[-1]:.3f}s, Acc@1 {top1_acc:.3f}, Acc@5 {top5_acc:.3f}')
    
    avg_batch_time = sum(batch_time) / len(batch_time)
    avg_top1 = sum(top1) / len(top1)
    avg_top5 = sum(top5) / len(top5)

    results = OrderedDict(
        model=args.model,
        top1=round(avg_top1, 4), top1_err=round(100 - avg_top1, 4),
        top5=round(avg_top5, 4), top5_err=round(100 - avg_top5, 4),
        img_size=args.img_size,
    )

    _logger.info(f' * Acc@1 {results["top1"]:.3f} ({results["top1_err"]:.3f}) Acc@5 {results["top5"]:.3f} ({results["top5_err"]:.3f})')

    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

def setup_default_logging():
    logging.basicConfig(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    main()
