import tensorflow as tf
import numpy as np
import math

class PolyLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Polynomial LR Scheduler with warmup, noise, and k-decay."""

    def __init__(self,
                 initial_learning_rate,
                 t_initial,
                 power=0.5,
                 lr_min=0.0,
                 cycle_mul=1.0,
                 cycle_decay=1.0,
                 cycle_limit=1,
                 warmup_t=0,
                 warmup_lr_init=0.0,
                 warmup_prefix=False,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 k_decay=1.0):
        super(PolyLRScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.t_initial = t_initial
        self.power = power
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.k_decay = k_decay
        self.rng = np.random.default_rng(seed=noise_seed)

        self.warmup_steps = []
        if warmup_t > 0:
            self.warmup_steps = [
                (self.initial_learning_rate - warmup_lr_init) / warmup_t
            ]
        else:
            self.warmup_steps = [1.0]

    def __call__(self, step):
        if self.t_in_epochs:
            return self._get_lr(step)
        return None

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = self.warmup_lr_init + t * self.warmup_steps[0]
        else:
            if self.warmup_prefix:
                t -= self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max = self.initial_learning_rate * gamma
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = self.lr_min + (lr_max - self.lr_min) * (1 - (t_curr ** k / t_i ** k)) ** self.power
            else:
                lrs = self.lr_min

        if self.noise_range_t and self.noise_range_t[0] <= t <= self.noise_range_t[1]:
            lrs += self._apply_noise(t)

        return lrs

    def _apply_noise(self, t):
        if self.noise_type == 'normal':
            return self.rng.normal(0, self.noise_std) * self.noise_pct
        elif self.noise_type == 'uniform':
            return self.rng.uniform(-self.noise_pct, self.noise_pct)
        return 0

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "t_initial": self.t_initial,
            "power": self.power,
            "lr_min": self.lr_min,
            "cycle_mul": self.cycle_mul,
            "cycle_decay": self.cycle_decay,
            "cycle_limit": self.cycle_limit,
            "warmup_t": self.warmup_t,
            "warmup_lr_init": self.warmup_lr_init,
            "warmup_prefix": self.warmup_prefix,
            "t_in_epochs": self.t_in_epochs,
            "noise_range_t": self.noise_range_t,
            "noise_pct": self.noise_pct,
            "noise_std": self.noise_std,
            "noise_seed": self.noise_seed,
            "k_decay": self.k_decay,
        }



