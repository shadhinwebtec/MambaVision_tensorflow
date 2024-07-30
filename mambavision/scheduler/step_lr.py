import tensorflow as tf
import numpy as np

class StepLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Step Learning Rate Scheduler with warmup and optional noise.
    """

    def __init__(self,
                 initial_learning_rate: float,
                 decay_t: float,
                 decay_rate: float = 1.0,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0.0,
                 t_in_epochs: bool = True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed: int = 42,
                 ) -> None:
        super(StepLRScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.rng = np.random.default_rng(seed=self.noise_seed)

        if self.warmup_t > 0:
            self.warmup_steps = [(self.initial_learning_rate - warmup_lr_init) / self.warmup_t]
        else:
            self.warmup_steps = [1.0]

    def __call__(self, step: int) -> float:
        if self.t_in_epochs:
            return self.get_epoch_values(step)
        else:
            return self.get_update_values(step)

    def get_epoch_values(self, epoch: int) -> float:
        if epoch < self.warmup_t:
            lr = self.warmup_lr_init + epoch * self.warmup_steps[0]
        else:
            lr = self.initial_learning_rate * (self.decay_rate ** (epoch // self.decay_t))
        return self._add_noise(lr, epoch)

    def get_update_values(self, num_updates: int) -> float:
        if num_updates < self.warmup_t:
            lr = self.warmup_lr_init + num_updates * self.warmup_steps[0]
        else:
            lr = self.initial_learning_rate * (self.decay_rate ** (num_updates // self.decay_t))
        return self._add_noise(lr, num_updates)

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
            'decay_t': self.decay_t,
            'decay_rate': self.decay_rate,
            'warmup_t': self.warmup_t,
            'warmup_lr_init': self.warmup_lr_init,
            't_in_epochs': self.t_in_epochs,
            'noise_range_t': self.noise_range_t,
            'noise_pct': self.noise_pct,
            'noise_std': self.noise_std,
            'noise_seed': self.noise_seed
        }
