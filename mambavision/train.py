import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from keras.preprocessing import image_dataset_from_directory
from keras.applications import EfficientNetB0
from keras.callbacks import TensorBoard, ModelCheckpoint

# Parsing arguments
config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser = argparse.ArgumentParser(description='TensorFlow ImageNet Training')

# Dataset parameters
parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='Input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=310, metavar='N', help='number of epochs to train (default: 310)')
parser.add_argument('--model', default='EfficientNetB0', type=str, metavar='MODEL', help='Name of model to train (default: "EfficientNetB0")')
parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N', help='number of label classes (default: 1000)')
parser.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder (default: none)')
parser.add_argument('--log_dir', default='./log_dir/', type=str, help='where to store tensorboard logs')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--amp', action='store_true', default=False, help='use mixed precision training')
parser.add_argument('--validate_only', action='store_true', default=False, help='run model validation only')

args = parser.parse_args()

# Set random seeds
tf.random.set_seed(args.seed)

# Data loading
train_dataset = image_dataset_from_directory(
    os.path.join(args.data_dir, 'train'),
    image_size=(224, 224),
    batch_size=args.batch_size,
    label_mode='int'
)

val_dataset = image_dataset_from_directory(
    os.path.join(args.data_dir, 'val'),
    image_size=(224, 224),
    batch_size=args.batch_size,
    label_mode='int'
)

# Model definition
if args.pretrained:
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
else:
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(args.num_classes, activation='softmax')
])

# Optimizer, loss, and metrics
optimizer = optimizers.Adam(learning_rate=args.learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# Compile model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Callbacks
callbacks = [
    TensorBoard(log_dir=args.log_dir),
    ModelCheckpoint(filepath=os.path.join(args.output, 'model_{epoch:02d}.h5'), save_best_only=True, monitor='val_loss', mode='min')
]

# Training
if not args.validate_only:
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
else:
    model.load_weights(args.resume)
    results = model.evaluate(val_dataset)
    print(f"Validation results: {results}")



from keras import mixed_precision
from keras.optimizers.schedules import LearningRateSchedule
from keras.callbacks import TensorBoard
from datetime import datetime
from absl import app, flags
import utils
from models import create_model
from dataset import create_daset 

# Placeholder imports for utility functions and custom models
# import utils
# from model import create_model
# from dataset import create_dataset

FLAGS = flags.FLAGS

def main(_):
    # Setup default logging
    utils.setup_default_logging()

    # Parse arguments
    args = FLAGS
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    strategy = tf.distribute.MultiWorkerMirroredStrategy() if args.distributed else tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync
    args.device = 'gpu' if tf.config.list_physical_devices('GPU') else 'cpu'
    print(f"Using {num_devices} devices: {args.device}")

    if args.amp:
        mixed_precision.set_global_policy('mixed_float16')
    else:
        mixed_precision.set_global_policy('float32')

    with strategy.scope():
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            attn_drop_rate=args.attn_drop_rate,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        # Load checkpoint if provided
        if args.resume:
            model.load_weights(args.resume)

        # Create datasets
        train_dataset, val_dataset = create_dataset(args.dataset, args.data_dir, args.batch_size, args.prefetcher)

        callbacks = []
        if args.rank == 0:
            log_dir = os.path.join(args.log_dir, args.tag)
            os.makedirs(log_dir, exist_ok=True)
            callbacks.append(TensorBoard(log_dir=log_dir))

        if args.model_ema:
            ema_callback = tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=args.ema_backup_dir)
            callbacks.append(ema_callback)

        model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )

        if args.validate_only:
            results = model.evaluate(val_dataset)
            print(f"Validation Results: {results}")
            return

        # Save final model
        if args.rank == 0:
            model.save(os.path.join(log_dir, 'final_model.h5'))

if __name__ == '__main__':
    # Define flags
    flags.DEFINE_string('model', 'resnet50', 'Model name')
    flags.DEFINE_string('data_dir', '/path/to/data', 'Directory containing dataset')
    flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
    flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
    flags.DEFINE_boolean('prefetcher', True, 'Use prefetcher for data loading')
    flags.DEFINE_float('lr', 0.001, 'Learning rate')
    flags.DEFINE_boolean('amp', False, 'Use automatic mixed precision')
    flags.DEFINE_string('log_dir', './logs', 'Directory for logging')
    flags.DEFINE_string('tag', '', 'Tag for logging')
    flags.DEFINE_boolean('validate_only', False, 'Only run validation')
    flags.DEFINE_string('resume', '', 'Resume training from checkpoint')
    flags.DEFINE_boolean('no_prefetcher', False, 'Disable prefetcher for data loading')
    flags.DEFINE_string('initial_checkpoint', '', 'Initial checkpoint for model')
    flags.DEFINE_float('drop_rate', 0.0, 'Dropout rate')
    flags.DEFINE_float('drop_path_rate', 0.0, 'Drop path rate')
    flags.DEFINE_integer('num_classes', None, 'Number of classes')
    flags.DEFINE_boolean('distributed', False, 'Enable distributed training')
    flags.DEFINE_float('bn_momentum', 0.9, 'Batch normalization momentum')
    flags.DEFINE_float('bn_eps', 1e-5, 'Batch normalization epsilon')
    flags.DEFINE_boolean('torchscript', False, 'Export model to TorchScript')
    flags.DEFINE_string('experiment', '', 'Experiment name')
    flags.DEFINE_boolean('no_saver', False, 'Disable model checkpoint saver')
    flags.DEFINE_string('output', './output', 'Output directory')
    flags.DEFINE_boolean('log_wandb', False, 'Log metrics to Weights & Biases')
    flags.DEFINE_string('ema_backup_dir', './ema_backup', 'Backup directory for EMA')

    app.run(main)



import tensorflow as tf
import time
from collections import OrderedDict

def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, lr_scheduler=None, saver=None, output_dir=None, mixed_precision_policy=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        if lr_scheduler is not None and not args.lr_ep:
            lr_scheduler.step_update(num_updates=(epoch * len(loader)) + batch_idx + 1)

        data_time_m.update(time.time() - end)
        
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        
        with tf.GradientTape() as tape:
            output = model(input, training=True)
            loss = loss_fn(target, output)

            if args.mesa > 0.0 and epoch / args.epochs > args.mesa_start_ratio:
                ema_output = model_ema(input, training=False)
                kd = kdloss(output, ema_output)
                loss += args.mesa * kd
        
        gradients = tape.gradient(loss, model.trainable_variables)
        if args.clip_grad is not None:
            gradients = [tf.clip_by_value(grad, -args.clip_grad, args.clip_grad) for grad in gradients]
        
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if model_ema is not None:
            model_ema.update(model)
        
        losses_m.update(loss.numpy(), input.shape[0])

        batch_time_m.update(time.time() - end)
        if batch_idx % args.log_interval == 0 or batch_idx == last_idx:
            print(f"Epoch {epoch} [{batch_idx}/{len(loader)}]  Loss: {losses_m.val:.4g} ({losses_m.avg:.3g})  Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s)")

        if saver is not None and args.recovery_interval and (batch_idx % args.recovery_interval == 0 or batch_idx == last_idx):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None and args.lr_ep:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
    
    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        if not args.prefetcher:
            input, target = input, target

        output = model(input, training=False)
        if isinstance(output, (tuple, list)):
            output = output[0]

        if args.tta > 1:
            output = tf.reduce_mean(tf.reshape(output, [-1, args.tta, output.shape[-1]]), axis=1)
            target = target[::args.tta]

        loss = loss_fn(target, output)
        acc1, acc5 = accuracy(output, target)

        losses_m.update(loss.numpy(), input.shape[0])
        top1_m.update(acc1.numpy(), input.shape[0])
        top5_m.update(acc5.numpy(), input.shape[0])

        batch_time_m.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0 or batch_idx == last_idx:
            print(f"Validate{log_suffix}: [{batch_idx}/{len(loader)}]  Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s)  Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  Acc@1: {top1_m.val:.4f} ({top1_m.avg:.4f})  Acc@5: {top5_m.val:.4f} ({top5_m.avg:.4f})")

    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

# Utility functions
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = tf.math.top_k(output, maxk, sorted=True)
    pred = tf.transpose(pred, perm=[1, 0])

    correct = tf.equal(pred, target)
    res = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(correct[:k], dtype=tf.float32))
        res.append(correct_k * (100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
