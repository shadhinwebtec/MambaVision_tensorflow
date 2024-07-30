import tensorflow as tf
import bisect
import numpy as np

class MultiStepLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate=1.0,
                 warmup_steps=0,
                 warmup_lr_init=0.0,
                 noise_range=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42):
        super(MultiStepLRScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.noise_range = noise_range
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(seed=noise_seed)

        if warmup_steps:
            self.warmup_increment = (initial_learning_rate - warmup_lr_init) / warmup_steps
        else:
            self.warmup_increment = 0

    def __call__(self, step):
        if step < self.warmup_steps:
            lr = self.warmup_lr_init + step * self.warmup_increment
        else:
            decay_steps = bisect.bisect_right(self.decay_steps, step + 1)
            lr = self.initial_learning_rate * (self.decay_rate ** decay_steps)

        if self.noise_range and self.noise_pct:
            if step >= self.noise_range[0] and step < self.noise_range[1]:
                noise = self.rng.normal(0, self.noise_std) * self.noise_pct
                lr += noise

        return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "warmup_steps": self.warmup_steps,
            "warmup_lr_init": self.warmup_lr_init,
            "noise_range": self.noise_range,
            "noise_pct": self.noise_pct,
            "noise_std": self.noise_std,
            "noise_seed": self.noise_seed,
        }
