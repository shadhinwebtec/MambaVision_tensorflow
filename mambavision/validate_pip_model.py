#!/usr/bin/env python3
""" ImageNet Validation Script using TensorFlow and Keras

This script validates pretrained models or training checkpoints against ImageNet or similarly organized datasets.
It uses standard TensorFlow/Keras practices for good performance. Modify as needed.

Hacked together by [Your Name] based on the original PyTorch script by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import logging
import os
import time
from collections import OrderedDict

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from keras.applications import (
    MobileNetV2,
    ResNet50,
    DenseNet121,
    InceptionV3,
)
from keras.preprocessing import image_dataset_from_directory
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Setup logging
_logger = logging.getLogger('validate')
logging.basicConfig(level=logging.INFO)

# Argument parser
parser = argparse.ArgumentParser(description='TensorFlow ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None, help='path to dataset')
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)')
parser.add_argument('--model', '-m', metavar='NAME', default='MobileNetV2', help='model architecture (default: MobileNetV2)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--img-size', default=224, type=int, metavar='N', help='input image dimension (default: 224)')
parser.add_argument('--log-freq', default=10, type=int, metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME', help='output json file for validation results')

def create_model(name):
    if name == 'MobileNetV2':
        return MobileNetV2(weights='imagenet')
    elif name == 'ResNet50':
        return ResNet50(weights='imagenet')
    elif name == 'DenseNet121':
        return DenseNet121(weights='imagenet')
    elif name == 'InceptionV3':
        return InceptionV3(weights='imagenet')
    else:
        raise ValueError(f"Unknown model name: {name}")

def load_dataset(data_dir, img_size, batch_size):
    return image_dataset_from_directory(
        data_dir,
        shuffle=False,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

def preprocess_fn(img, label):
    return preprocess_input(img), label

def validate(args):
    device = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"
    
    with tf.device(device):
        model = create_model(args.model)

        if args.checkpoint:
            model.load_weights(args.checkpoint)
            _logger.info(f'Loaded model weights from {args.checkpoint}')
        
        dataset = load_dataset(args.data_dir, args.img_size, args.batch_size)
        dataset = dataset.map(preprocess_fn)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        top1_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1')
        top5_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')

        losses = []
        top1_scores = []
        top5_scores = []
        batch_time = []

        for step, (images, labels) in enumerate(dataset):
            start_time = time.time()
            logits = model(images, training=False)
            loss = loss_fn(labels, logits)
            top1_metric.update_state(labels, logits)
            top5_metric.update_state(labels, logits)
            end_time = time.time()

            losses.append(loss.numpy())
            top1_scores.append(top1_metric.result().numpy())
            top5_scores.append(top5_metric.result().numpy())
            batch_time.append(end_time - start_time)

            if step % args.log_freq == 0:
                _logger.info(
                    f'Step {step}: Loss = {loss.numpy()}, Top1 Acc = {top1_metric.result().numpy()}, '
                    f'Top5 Acc = {top5_metric.result().numpy()}, Time = {batch_time[-1]}s'
                )

        results = OrderedDict(
            model=args.model,
            top1=round(np.mean(top1_scores), 4),
            top1_err=round(100 - np.mean(top1_scores), 4),
            top5=round(np.mean(top5_scores), 4),
            top5_err=round(100 - np.mean(top5_scores), 4),
            param_count=model.count_params() / 1e6,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )

        _logger.info(f' * Acc@1 {results["top1"]:.3f} ({results["top1_err"]:.3f}) Acc@5 {results["top5"]:.3f} ({results["top5_err"]:.3f})')

        return results

def main():
    args = parser.parse_args()
    results = validate(args)

    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=4)
        _logger.info(f'Results saved to {args.results_file}')

    print(f'--result\n{json.dumps(results, indent=4)}')

if __name__ == '__main__':
    main()
