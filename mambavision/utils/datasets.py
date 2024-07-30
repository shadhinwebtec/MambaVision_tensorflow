import os
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing import image_dataset_from_directory
from keras.applications import imagenet_utils
from sklearn.model_selection import train_test_split

# Helper function to apply transforms
def preprocess_image(image, label, input_size, train=False):
    image = tf.image.resize(image, [input_size, input_size])
    image = tf.image.convert_image_dtype(image, tf.float32)
    if train:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = imagenet_utils.preprocess_input(image * 255.0)
    return image, label

# Function to get dataset loaders
def get_loaders(args, mode='eval', dataset=None):
    if dataset is None:
        dataset = args.dataset

    if dataset == 'imagenet':
        return get_imagenet_loader(args, mode)
    else:
        if mode == 'search':
            return get_loaders_search(args)
        elif mode == 'eval':
            return get_loaders_eval(dataset, args)

# Function for CIFAR-10 and CIFAR-100
def get_loaders_eval(dataset, args):
    input_size = args.resolution
    batch_size = args.batch_size
    data_dir = args.data

    if dataset == 'cifar10':
        (train_images, train_labels), (valid_images, valid_labels) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif dataset == 'cifar100':
        (train_images, train_labels), (valid_images, valid_labels) = tf.keras.datasets.cifar100.load_data()
        num_classes = 100

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(lambda x, y: preprocess_image(x, y, input_size, train=True)).shuffle(buffer_size=50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
    valid_ds = valid_ds.map(lambda x, y: preprocess_image(x, y, input_size)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, num_classes

# Function for data loading in search mode
def get_loaders_search(args):
    input_size = args.resolution
    batch_size = args.batch_size
    data_dir = args.data

    if args.dataset == 'cifar10':
        (images, labels), _ = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif args.dataset == 'cifar100':
        (images, labels), _ = tf.keras.datasets.cifar100.load_data()
        num_classes = 100

    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=1-args.train_portion, random_state=0)
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.map(lambda x, y: preprocess_image(x, y, input_size, train=True)).shuffle(buffer_size=len(train_images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_images, valid_labels))
    valid_ds = valid_ds.map(lambda x, y: preprocess_image(x, y, input_size)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, num_classes

# Function for ImageNet data loading
def get_imagenet_loader(args, mode='eval', testdir=""):
    input_size = args.resolution
    batch_size = args.batch_size
    data_dir = args.data

    if len(testdir) < 2:
        testdir = os.path.join("../ImageNetV2/", 'test')

    if mode == 'eval':
        train_dir = os.path.join(data_dir, 'train')
        valid_dir = os.path.join(data_dir, 'val')

        train_ds = image_dataset_from_directory(train_dir, image_size=(input_size, input_size), batch_size=batch_size)
        valid_ds = image_dataset_from_directory(valid_dir, image_size=(input_size, input_size), batch_size=batch_size)

        train_ds = train_ds.map(lambda x, y: preprocess_image(x, y, input_size, train=True)).prefetch(tf.data.AUTOTUNE)
        valid_ds = valid_ds.map(lambda x, y: preprocess_image(x, y, input_size)).prefetch(tf.data.AUTOTUNE)

    elif mode == 'search':
        train_dir = os.path.join(data_dir, 'train')

        all_data = image_dataset_from_directory(train_dir, image_size=(input_size, input_size), batch_size=batch_size)
        train_data, valid_data = train_test_split(all_data, test_size=1-args.train_portion, random_state=0)

        train_ds = train_data.map(lambda x, y: preprocess_image(x, y, input_size, train=True)).prefetch(tf.data.AUTOTUNE)
        valid_ds = valid_data.map(lambda x, y: preprocess_image(x, y, input_size)).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, 1000

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Data loading for evaluation and search')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name')
    parser.add_argument('--data', type=str, default='/data/datasets/imagenet', help='Dataset path')
    parser.add_argument('--train_portion', type=float, default=0.9, help='Portion of training data')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--resolution', type=int, default=224, help='Input resolution size')
    args = parser.parse_args()

    train_ds, valid_ds, _ = get_imagenet_loader(args, mode='search')

    for batch in train_ds.take(1):
        images, labels = batch
        print(images.shape, labels.shape)

    for batch in valid_ds.take(1):
        images, labels = batch
        print(images.shape, labels.shape)
