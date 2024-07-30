import tensorflow as tf
import numpy as np
from typing import Dict, Any

class Scheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    This is intended to be called:
    * At the END of each epoch, before incrementing the epoch count, to calculate the next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate the next update's value
    """

    def __init__(self,
                 initial_learning_rate: float,
                 param_group_field: str = 'lr',
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        super(Scheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.param_group_field = param_group_field
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.rng = np.random.default_rng(seed=self.noise_seed)
        self.base_values = initial_learning_rate
        self.metric = None

        if initialize:
            self.update_groups(self.base_values)

    def __call__(self, step: int) -> float:
        if self.noise_range_t is not None and self._is_apply_noise(step):
            lr = self._add_noise([self.initial_learning_rate], step)[0]
        else:
            lr = self.initial_learning_rate
        return lr

    def get_config(self) -> Dict[str, Any]:
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'param_group_field': self.param_group_field,
            'noise_range_t': self.noise_range_t,
            'noise_pct': self.noise_pct,
            'noise_type': self.noise_type,
            'noise_std': self.noise_std,
            'noise_seed': self.noise_seed,
        }

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def get_epoch_values(self, epoch: int):
        # Override in child class
        return None

    def get_update_values(self, num_updates: int):
        # Override in child class
        return None

    def update_groups(self, values):
        # Here, values would be used to set the learning rates or other params
        self.base_values = values

    def _add_noise(self, lrs, t):
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lrs = [v + v * noise for v in lrs]
        return lrs

    def _is_apply_noise(self, t) -> bool:
        """Return True if scheduler is in noise range."""
        apply_noise = False
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
        return apply_noise

    def _calculate_noise(self, t) -> float:
        self.rng = np.random.default_rng(seed=self.noise_seed + t)
        if self.noise_type == 'normal':
            while True:
                # Resample if noise out of percent limit, brute force but shouldn't spin much
                noise = self.rng.normal(0, self.noise_std)
                if abs(noise) < self.noise_pct:
                    return noise
        else:
            noise = 2 * (self.rng.uniform() - 0.5) * self.noise_pct
        return noise
