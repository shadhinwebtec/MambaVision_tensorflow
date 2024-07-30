import tensorflow as tf
import numpy as np

class TanhLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Hyperbolic-Tangent decay with restarts, warmup, and optional noise.
    """

    def __init__(self,
                 initial_learning_rate: float,
                 t_initial: int,
                 lb: float = -7.0,
                 ub: float = 3.0,
                 lr_min: float = 0.0,
                 cycle_mul: float = 1.0,
                 cycle_decay: float = 1.0,
                 cycle_limit: int = 1,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0.0,
                 warmup_prefix: bool = False,
                 t_in_epochs: bool = True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed: int = 42,
                 ) -> None:
        super(TanhLRScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.t_initial = t_initial
        self.lb = lb
        self.ub = ub
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
        self.rng = np.random.default_rng(seed=self.noise_seed)

        if self.warmup_t > 0:
            t_v = [initial_learning_rate] if self.warmup_prefix else self._get_lr(self.warmup_t)
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in t_v]
        else:
            self.warmup_steps = [1.0]

    def __call__(self, step: int) -> float:
        if self.t_in_epochs:
            return self.get_epoch_values(step)
        else:
            return self.get_update_values(step)

    def get_epoch_values(self, epoch: int) -> float:
        return self._get_lr(epoch)

    def get_update_values(self, num_updates: int) -> float:
        return self._get_lr(num_updates)

    def _get_lr(self, t: int) -> float:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = np.floor(np.log(1 - t / self.t_initial * (1 - self.cycle_mul)) / np.log(self.cycle_mul)).astype(int)
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            if i < self.cycle_limit:
                gamma = self.cycle_decay ** i
                lr_max = self.initial_learning_rate * gamma

                tr = t_curr / t_i
                lr = self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 - np.tanh(self.lb * (1. - tr) + self.ub * tr))
            else:
                lr = self.lr_min
        return self._add_noise(lr, t)

    def _add_noise(self, lr: float, t: int) -> float:
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lr += lr * noise
        return lr

    def _is_apply_noise(self, t: int) -> bool:
        """Return True if scheduler is in noise range."""
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                return self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                return t >= self.noise_range_t
        return False

    def _calculate_noise(self, t: int) -> float:
        self.rng = np.random.default_rng(seed=self.noise_seed + t)
        if self.noise_std > 0:
            if self.noise_pct > 0:
                while True:
                    noise = self.rng.normal(0, self.noise_std)
                    if abs(noise) < self.noise_pct:
                        return noise
            else:
                return self.rng.normal(0, self.noise_std)
        return 0.0

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            't_initial': self.t_initial,
            'lb': self.lb,
            'ub': self.ub,
            'lr_min': self.lr_min,
            'cycle_mul': self.cycle_mul,
            'cycle_decay': self.cycle_decay,
            'cycle_limit': self.cycle_limit,
            'warmup_t': self.warmup_t,
            'warmup_lr_init': self.warmup_lr_init,
            'warmup_prefix': self.warmup_prefix,
            't_in_epochs': self.t_in_epochs,
            'noise_range_t': self.noise_range_t,
            'noise_pct': self.noise_pct,
            'noise_std': self.noise_std,
            'noise_seed': self.noise_seed
        }
