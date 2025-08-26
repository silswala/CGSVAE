from math import gamma

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from Utility.mylogging import setup_logging
from Utility.config_handler import get_nested_config

from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, LambdaLR,
    ReduceLROnPlateau, CyclicLR, OneCycleLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts
)


class Optimizers:
    """Manages optimizers and learning rate schedulers.

    Args:
        config (dict): Configuration dictionary containing "optimizer", "learning_rate",
                       and optionally "scheduler" with scheduler-specific parameters.
        model (torch.nn.Module): The neural network model.

    Raises:
        ValueError: If an unsupported optimizer or scheduler is specified,
                    or if required configuration parameters are missing.
    """

    def __init__(self, config: dict, model: torch.nn.Module):
        """Initializes the Optimizers object.

        Args:
            config (dict): Optimizer and scheduler configuration.
            model (torch.nn.Module): The neural network model.
        """
        logdir = get_nested_config(config, ["run_info", "log_file_name"])
        self.logger = setup_logging(log_file_path=logdir, logger_object_name="sepcvaeutil")
        self._optimizer = get_nested_config(config, ["training", "optimizer"])
        self._learning_rate = get_nested_config(config, ["training", "learning_rate"])
        self._config = config
        self._model = model

        self.logger.info(f"Optimizer initialized with {self._optimizer} optimizer.")

    def setup(self):
        """Sets up the optimizer based on the configuration.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.

        Raises:
            ValueError: If the specified optimizer is not supported.
        """
        if self._optimizer == "adam":
            _optim = optim.Adam(self._model.parameters(), lr=self._learning_rate)
            self.logger.info("Using Adam optimizer.")
            return _optim
        else:
            self.logger.error(f"Optimizer '{self._optimizer}' not supported.")
            raise ValueError(f"Optimizer '{self._optimizer}' not supported.")

    def get_scheduler(self, optimizer):
        """Sets up the learning rate scheduler based on the configuration.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to which the scheduler
                                             will be attached.

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The initialized scheduler,
                                                                or None if no scheduler is specified.

        Raises:
            ValueError: If the specified scheduler is not supported or if required
                        scheduler parameters are missing from the configuration.
        """
        if "scheduler" not in self._config["training"]:
            self.logger.warning("No scheduler specified in config.")
            return None  # Return None explicitly when no scheduler is configured

        scheduler_type = get_nested_config(self._config, ["training", "scheduler", "name"])

        if scheduler_type == "steplr":

            step_size = get_nested_config(self._config, ["training", "scheduler", "step_size"])
            gamma = get_nested_config(self._config, ["training", "scheduler", "gamma"])

            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            self.logger.info(f"Using StepLR scheduler with step_size={step_size} and gamma={gamma}.")
            return scheduler
        # ... other scheduler types ...
        else:
            error_message = f"Scheduler '{scheduler_type}' not supported."
            self.logger.error(error_message)
            raise ValueError(error_message)


def get_scheduler_prototype(self, optimizer):
    """Sets up the learning rate scheduler based on the configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to which the scheduler
                                           will be attached.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: The initialized scheduler,
                                                        or None if no scheduler is specified.

    Raises:
        ValueError: If the specified scheduler is not supported or if required
                    scheduler parameters are missing from the configuration.
    """
    if "scheduler" not in self._config["training"]:
        self.logger.warning("No scheduler specified in config.")
        return None  # Return None explicitly when no scheduler is configured

    scheduler_type = get_nested_config(self._config, ["training", "scheduler", "name"])
    self.logger.info(f"Initializing scheduler: {scheduler_type}")

    if scheduler_type == "steplr":
        step_size = get_nested_config(self._config, ["training", "scheduler", "step_size"])
        gamma = get_nested_config(self._config, ["training", "scheduler", "gamma"], default=0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "multisteplr":
        milestones = get_nested_config(self._config, ["training", "scheduler", "milestones"])
        gamma = get_nested_config(self._config, ["training", "scheduler", "gamma"], default=0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == "exponentiallr":
        gamma = get_nested_config(self._config, ["training", "scheduler", "gamma"], default=0.99)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == "lambdalr":
        lambda_fn = get_nested_config(self._config, ["training", "scheduler", "lambda_fn"])
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)

    elif scheduler_type == "reducelronplateau":
        mode = get_nested_config(self._config, ["training", "scheduler", "mode"], default="min")
        factor = get_nested_config(self._config, ["training", "scheduler", "factor"], default=0.1)
        patience = get_nested_config(self._config, ["training", "scheduler", "patience"], default=10)
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

    elif scheduler_type == "cycliclr":
        base_lr = get_nested_config(self._config, ["training", "scheduler", "base_lr"])
        max_lr = get_nested_config(self._config, ["training", "scheduler", "max_lr"])
        step_size_up = get_nested_config(self._config, ["training", "scheduler", "step_size_up"])
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular')

    elif scheduler_type == "onecyclelr":
        max_lr = get_nested_config(self._config, ["training", "scheduler", "max_lr"])
        total_steps = get_nested_config(self._config, ["training", "scheduler", "total_steps"])
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)

    elif scheduler_type == "cosineannealinglr":
        T_max = get_nested_config(self._config, ["training", "scheduler", "T_max"])
        eta_min = get_nested_config(self._config, ["training", "scheduler", "eta_min"], default=0)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == "cosineannealingwarmrestarts":
        T_0 = get_nested_config(self._config, ["training", "scheduler", "T_0"])
        T_mult = get_nested_config(self._config, ["training", "scheduler", "T_mult"], default=1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    else:
        error_message = f"Scheduler '{scheduler_type}' not supported."
        self.logger.error(error_message)
        raise ValueError(error_message)

    self.logger.info(f"Using {scheduler_type} scheduler.")
    return scheduler
