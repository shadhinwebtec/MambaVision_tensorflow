import tensorflow as tf
import numpy as np

class PlateauLRScheduler(tf.keras.callbacks.Callback):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,
                 decay_rate=0.1,
                 patience_t=10,
                 verbose=True,
                 threshold=1e-4,
                 cooldown_t=0,
                 warmup_t=0,
                 warmup_lr_init=0,
                 lr_min=0,
                 mode='max',
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 ):
        super(PlateauLRScheduler, self).__init__()
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.patience_t = patience_t
        self.verbose = verbose
        self.threshold = threshold
        self.cooldown_t = cooldown_t
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.lr_min = lr_min
        self.mode = mode
        self.noise_range_t = noise_range_t
        self.noise_type = noise_type
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(seed=noise_seed)

        self.best = -np.inf if mode == 'max' else np.inf
        self.best_epoch = 0
        self.wait = 0
        self.cooldown_counter = 0
        self.restore_lr = None

        self.warmup_steps = []
        if self.warmup_t > 0:
            self.warmup_steps = [
                (self.optimizer.lr - warmup_lr_init) / warmup_t for _ in self.optimizer.weights
            ]
            self._update_lr(warmup_lr_init)
        else:
            self.warmup_steps = [0 for _ in self.optimizer.weights]

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss') if self.mode == 'min' else logs.get('val_accuracy')
        if epoch < self.warmup_t:
            new_lr = self.warmup_lr_init + epoch * np.mean(self.warmup_steps)
            self._update_lr(new_lr)
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0

            if self._is_improvement(current):
                self.best = current
                self.best_epoch = epoch
                self.wait = 0
            elif not self.in_cooldown:
                self.wait += 1
                if self.wait >= self.patience_t:
                    if self.verbose:
                        print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {self.optimizer.lr * self.decay_rate}.")
                    self._reduce_lr()
                    self.cooldown_counter = self.cooldown_t
                    self.wait = 0

            if self.noise_range_t and self.noise_range_t[0] <= epoch <= self.noise_range_t[1]:
                self._apply_noise(epoch)

    def _is_improvement(self, current):
        if self.mode == 'max':
            return current > self.best + self.threshold
        else:
            return current < self.best - self.threshold

    def _reduce_lr(self):
        old_lr = float(self.optimizer.lr)
        new_lr = max(old_lr * self.decay_rate, self.lr_min)
        self.optimizer.lr.assign(new_lr)
        self.restore_lr = old_lr

    def _apply_noise(self, epoch):
        noise = self._calculate_noise(epoch)
        old_lr = float(self.optimizer.lr)
        new_lr = old_lr + old_lr * noise
        self.restore_lr = old_lr
        self.optimizer.lr.assign(new_lr)

    def _calculate_noise(self, epoch):
        if self.noise_type == 'normal':
            return self.rng.normal(0, self.noise_std) * self.noise_pct
        elif self.noise_type == 'uniform':
            return self.rng.uniform(-self.noise_pct, self.noise_pct)
        return 0

    def _update_lr(self, new_lr):
        self.optimizer.lr.assign(new_lr)

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
