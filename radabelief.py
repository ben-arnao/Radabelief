import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typing import Union, Callable, Dict
from typeguard import typechecked
import numpy as np


@tf.keras.utils.register_keras_serializable(package="Addons")
class RadaBelief(tf.keras.optimizers.Optimizer):

    @typechecked
    def __init__(
            self,
            learning_rate: Union[FloatTensorLike, Callable, Dict] = 1e-4,
            beta_1: FloatTensorLike = 0.9,
            beta_2: FloatTensorLike = 0.999,
            epsilon: FloatTensorLike = 1e-12,
            warmup_steps: int = 10000,
            name: str = "Radabelief",
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("warmup_steps", warmup_steps)
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype

        # *current* learn rate of optimizer (changed by ReduceLROnPlateau for example)
        lr_t = self.lr

        # get previous moments
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # static hyperparameters of optimizer
        lr_0 = self._get_hyper("learning_rate", var_dtype)
        warmup_steps = self._get_hyper("warmup_steps", var_dtype)
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)

        # smaller epsilon == more bias == more like SGD
        # larger epsilon == more adaptive, potential for large LR difference between variables
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)

        # current step of optimizer
        local_step = tf.cast(self.iterations + 1, var_dtype)

        # use linearly scaled lr from 0 to initial LR while steps under 'warmup_steps'
        lr = tf.where(
            local_step <= warmup_steps,
            (local_step / warmup_steps) * lr_0,
            lr_t,
        )

        # calculate first moment of gradient (momemtum)
        m_t = m.assign(
            beta_1 * m + (1.0 - beta_1) * grad,
            use_locking=self._use_locking
        )

        # calculate second moment of gradient (RMSprop)
        # use 'tf.square(grad - m_t)' for Adabelief instead of 'tf.square(grad)'
        v_t = v.assign(
            beta_2 * v + (1.0 - beta_2) * tf.square(grad - m_t),
            use_locking=self._use_locking,
        )

        # correct bias (mostly affects initial steps)
        m_corr_t = m_t / (1.0 - tf.pow(beta_1, local_step))
        v_corr_t = v_t / (1.0 - tf.pow(beta_2, local_step))

        # calculate step
        var_t = m_corr_t / (tf.sqrt(v_corr_t) + epsilon_t)

        # apply learn rate
        var_update = var.assign_sub(lr * var_t,
                                    use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "warmup_steps": self._serialize_hyperparameter("warmup_steps"),
            }
        )
        return config
