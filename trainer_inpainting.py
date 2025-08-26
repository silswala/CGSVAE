import sys
import random

import numpy as np
import torch.nn
from numpy.ma.extras import average
import matplotlib.pyplot as plt
from DataManager.data_loader import *
from DataManager.spectrum_dataset import SpectraDataset
from Utility.mylogging import *
from Utility.config_handler import get_nested_config, parse_config_json
import sqlite3
from Utility.specvaeutil import Optimizers
from Utility.dbhandler import DBHandler
from torch.utils.data import DataLoader
import time

from models.loss import *
import os  # Import the os module for path manipulation


def contiguous_masking_gpu_efficient(tensor, mask_length_ratio, min_num_masks, max_num_masks):
	"""Applies contiguous masks (sets blocks of elements to zero) to a tensor,
	ensuring operations occur on the tensor's device. The number of masks is
	randomly determined within a specified range.

	Args:
		tensor: The input tensor (can be on CPU or GPU).
		mask_length_ratio: The proportion of the tensor's total length (last dimension)
							that each contiguous mask should cover (between 0 and 1).
		min_num_masks: The minimum number of contiguous blocks to mask (inclusive).
		max_num_masks: The maximum number of contiguous blocks to mask (inclusive).

	Returns:
		A new tensor with masked elements, on the same device as the input tensor.

	Raises:
		ValueError: If mask_length_ratio is not between 0.0 and 1.0, or if
					min_num_masks or max_num_masks are not positive integers, or
					if min_num_masks > max_num_masks.
	"""
	if not (0.0 <= mask_length_ratio <= 1.0):
		raise ValueError("mask_length_ratio must be between 0.0 and 1.0.")
	if not isinstance(min_num_masks, int) or min_num_masks <= 0:
		raise ValueError("min_num_masks must be a positive integer.")
	if not isinstance(max_num_masks, int) or max_num_masks <= 0:
		raise ValueError("max_num_masks must be a positive integer.")
	if min_num_masks > max_num_masks:
		raise ValueError("min_num_masks cannot be greater than max_num_masks.")

	# Randomly determine the number of masks within the given range
	num_masks = torch.randint(min_num_masks, max_num_masks + 1, (1,), device=tensor.device).item()

	# Create a copy of the tensor to modify, ensuring it's on the correct device
	masked_tensor = tensor.clone().to(tensor.device)

	# Determine the length of the dimension to mask (usually the last for sequences/spectrums)
	dim_to_mask = -1
	tensor_length = tensor.shape[dim_to_mask]

	# Calculate the actual length of each mask
	mask_actual_length = int(tensor_length * mask_length_ratio)

	# Ensure mask_actual_length is at least 1, unless ratio is 0
	if mask_length_ratio > 0 and mask_actual_length == 0:
		mask_actual_length = 1

	if mask_actual_length == 0:  # If mask_length_ratio is 0, no masking
		return masked_tensor

	# Iterate to apply the specified number of masks
	for _ in range(num_masks):
		# 1. Randomly choose a starting index for the mask
		start_idx_limit = tensor_length - mask_actual_length
		if start_idx_limit < 0:  # Mask is longer than tensor
			start_idx = 0
			end_idx = tensor_length
		else:
			# Generate start_idx on the correct device
			start_idx = torch.randint(0, start_idx_limit + 1, (1,), device=tensor.device).item()
			end_idx = start_idx + mask_actual_length

		# 2. Apply the mask (set elements to zero)
		slices = [slice(None)] * masked_tensor.ndim
		slices[dim_to_mask] = slice(start_idx, end_idx)

		masked_tensor[tuple(slices)] = 0.0

	return masked_tensor


def random_masking_gpu_efficient(tensor, mask_ratio):
	"""Randomly masks elements in a tensor, ensuring operations occur on the tensor's device.

	Args:
		tensor: The input tensor (can be on CPU or GPU).
		mask_ratio: The proportion of elements to mask (set to zero) (between 0 and 1).

	Returns:
		A new tensor with masked elements, on the same device as the input tensor.
	"""
	if not (0.0 <= mask_ratio <= 1.0):
		raise ValueError("mask_ratio must be between 0.0 and 1.0.")

	# 1. Generate random numbers on the same device as the input tensor.
	random_numbers = torch.rand(tensor.shape, device=tensor.device)

	# 2. Create a boolean mask. 'True' where elements should be masked (set to zero).
	mask = random_numbers < mask_ratio

	# 3. Use torch.where to select elements.
	masked_tensor = torch.where(mask, torch.zeros_like(tensor), tensor)

	return masked_tensor


def apply_random_masking_strategy(tensor,
								  mask_type_probabilities={'contiguous': 0.5, 'random': 0.5},
								  contiguous_mask_length_ratio=0.1,
								  contiguous_min_num_masks=1,
								  contiguous_max_num_masks=3,
								  random_mask_ratio=0.1):
	"""Applies either contiguous or random masking to a tensor based on probabilities.

	Args:
		tensor: The input tensor (can be on CPU or GPU).
		mask_type_probabilities: A dictionary specifying the probability of
								 choosing each masking type. Keys should be
								 'contiguous' and 'random'. Values should sum to 1.0.
		contiguous_mask_length_ratio: For contiguous masking, the proportion of
									  the tensor's length each mask covers.
		contiguous_min_num_masks: For contiguous masking, the minimum number of masks.
		contiguous_max_num_masks: For contiguous masking, the maximum number of masks.
		random_mask_ratio: For random masking, the proportion of elements to mask.

	Returns:
		A new tensor with masked elements, on the same device as the input tensor.

	Raises:
		ValueError: If mask_type_probabilities are invalid, or if any
					masking-specific arguments are invalid (propagated from
					individual masking functions).
	"""
	mask_types = list(mask_type_probabilities.keys())
	probabilities = list(mask_type_probabilities.values())

	if not (all(isinstance(p, (float, int)) and p >= 0 for p in probabilities) and
			abs(sum(probabilities) - 1.0) < 1e-6):
		raise ValueError("mask_type_probabilities values must be non-negative and sum to 1.0.")

	# Randomly choose a masking type based on provided probabilities
	chosen_mask_type = random.choices(mask_types, weights=probabilities, k=1)[0]

	if chosen_mask_type == 'contiguous':
		return contiguous_masking_gpu_efficient(tensor,
												contiguous_mask_length_ratio,
												contiguous_min_num_masks,
												contiguous_max_num_masks)
	elif chosen_mask_type == 'random':
		return random_masking_gpu_efficient(tensor, random_mask_ratio)
	else:
		# This case should ideally not be reached if mask_types are correctly handled
		raise ValueError(f"Unknown mask type chosen: {chosen_mask_type}")

def compute_max_norm(model, c=0.5):
    """
    Compute max gradient norm dynamically based on model depth.

    Parameters:
    - model: PyTorch model
    - c: Scaling constant (default: 0.5)

    Returns:
    - max_norm: Computed max_norm for gradient clipping
    """
    L = sum(1 for _ in model.parameters() if _.requires_grad)  # Count layers with gradients
    max_norm = c / (L ** 0.5)
    return max_norm

class InpaintingTrainer:
    """
    A class to manage the training and evaluation of a CGSVAE model.

    Args:
        model (torch.nn.Module): The CGSVAE model to train.
        config (dict): A dictionary containing the training configuration.
    """

    def __init__(self, model: torch.nn.Module, config):
        #
        # Setting up logging
        #
        self.logdir = get_nested_config(config=config, keys=["run_info","log_file_name"])
        self.logger = setup_logging(log_file_path=self.logdir, logger_object_name="InCGSVAE Trainer", level=logging.DEBUG)

        self.model = model
        self.device = get_nested_config(config, ["device"])
        self.additional_output_dim = get_nested_config(config, ["model", "additional_output_dim"])
        self.train_db_names = get_nested_config(config, ["data", "train_db_names"])
        self.batch_size = get_nested_config(config, ["training", "batch_size"])
        self.shuffle_train_data = get_nested_config(config, ["training", "shuffle_train_data"])
        self.test_db_names = get_nested_config(config, ["data", "test_db_names"])
        self.shuffle_test_data = get_nested_config(config, ["training", "shuffle_test_data"])
        self.epoch = get_nested_config(config, ["training", "epoch"])
        self.losses = get_nested_config(config, ["training", "losses"])
        self.training_batch_per_epoch_log_freq = get_nested_config(config, ["training", "logging", "training_batch_per_epoch_log_freq"])
        self.start_epoch_checkpoint = get_nested_config(config, ["training", "logging", "start_epoch_checkpoint"])
        self.checkpoint_freq = get_nested_config(config, ["training", "logging", "checkpoint_freq"])
        self.run_id = get_nested_config(config, ["run_info", "run_id"])
        self.check_points = get_nested_config(config, ["run_info", "check_points"])
        self.config = config
        self.train_loss_table_name = get_nested_config(config, ["train_loss_table_name"])
        self.test_loss_table_name = get_nested_config(config, ["test_loss_table_name"])
        self.db_save_frequency = get_nested_config(config, ["db_save_frequency"])
        self.loss_max = 200000
        self.is_loss_clip = False
        # self.gradient_clip_max = compute_max_norm(self.model, c=0.9)
        self.gradient_clip_max = 10
        self.logger.info(f"Max norm is set as: {self.gradient_clip_max}")
        self.is_gradient_clip = True
        self.masking_strategy_contiguous_prob = get_nested_config(config, ["masking", "masking_strategy", "contiguous"])
        self.masking_strategy_random_prob = get_nested_config(config, ["masking", "masking_strategy", "random"])
        self.contiguous_mask_length_ratio = get_nested_config(config, ["masking", "contiguous_mask_length_ratio"])
        self.contiguous_min_num_masks = get_nested_config(config, ["masking", "contiguous_min_num_masks"])
        self.contiguous_max_num_masks = get_nested_config(config, ["masking", "contiguous_max_num_masks"])
        self.random_mask_ratio = get_nested_config(config, ["masking", "random_mask_ratio"])


        #
        # Loading train and test data
        #
        self.logger.info("Loading train data ...")
        _x_train, _y_train = load_data_from_multiple_dbs(self.train_db_names)
        # self.logger.info(_y_train[:5])
        # print(_y_train[:5])
        # print(f"Sum of Data : {_x_train[10]}")

        # Log the length of the raw data (before dataset creation)
        self.logger.info(
            f"Loaded {_x_train.shape[0] if hasattr(_x_train, 'shape') else len(_x_train)} training samples.")

        arr_labels_size = _y_train[0].shape[0] if hasattr(_y_train[0], 'shape') else len(_y_train[0])

        if "rn" in self.losses and self.additional_output_dim != arr_labels_size:
            error_msg = f"""The loss calculation is requesting rn loss, but 'additional_output_dim' length ({self.additional_output_dim}) is not equal to the size of the labels array is {arr_labels_size}"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if self.additional_output_dim>0 and self.additional_output_dim != arr_labels_size:
            error_msg = f"""The additional_output_dim is set {self.additional_output_dim} but the size of the labels array is {arr_labels_size}"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Train data loading completed, creating dataset...")
        self.train_dataset = SpectraDataset(spectra_list=_x_train, labels_list=_y_train)

        # Log the length of the dataset
        self.train_dataset_size = len(self.train_dataset)
        self.logger.info(f"Train dataset created with {self.train_dataset_size} samples.")

        
        if self.train_dataset_size == 0:  # Check for empty dataset
            self.logger.error("Train dataset is empty! Check your data loading process.")
            raise ValueError("Train dataset is empty.")

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                          shuffle=self.shuffle_train_data)

        self.logger.info("Train data loaded!")

        self.logger.info("Loading test data ...")
        _x_test, _y_test = load_data_from_multiple_dbs(self.test_db_names)

        # Log the length of the raw test data
        self.logger.info(
            f"Loaded {_x_test.shape[0] if hasattr(_x_test, 'shape') else len(_x_test)} test samples.")

        self.logger.info("Test data loading completed, creating dataset...")
        self.test_dataset = SpectraDataset(spectra_list=_x_test, labels_list=_y_test)
        self.test_dataset_size = len(self.test_dataset)

        # Log the length of the test dataset
        self.logger.info(f"Test dataset created with {self.test_dataset_size} samples.")

        if self.test_dataset_size == 0:  # Check for empty dataset
            self.logger.error("Test dataset is empty! Check your data loading process.")
            raise ValueError("Test dataset is empty.")

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                         shuffle=self.shuffle_test_data)
        self.logger.info("Test data loaded!")

    def train_model(self):
        """
        Trains the spectral VAE model with epoch-level timing.
        """
        self.logger.info(horizontal_line(length=100, space=0, char="."))
        self.logger.info("Training started ...")

        # Optimizer and scheduler setup
        start_optimizer_setup = time.time()
        optimizer_instance = Optimizers(self.config, self.model)
        optimizer = optimizer_instance.setup()
        scheduler = optimizer_instance.get_scheduler(optimizer=optimizer)
        end_optimizer_setup = time.time()
        self.logger.info(f"Optimizer setup took: {end_optimizer_setup - start_optimizer_setup:.4f} seconds.")

        # DB handler setup
        start_db_handler_setup = time.time()
        db_handler = DBHandler(self.config)
        db_handler.create_losses_table(self.train_loss_table_name)
        db_handler.create_losses_table(self.test_loss_table_name)
        end_db_handler_setup = time.time()
        self.logger.info(f"DB handler setup took: {end_db_handler_setup - start_db_handler_setup:.4f} seconds.")

        self.logger.info(horizontal_line(50, 2, "-"))

        train_loss_data_buffer = []
        test_loss_data_buffer = []

        for epoch in range(self.epoch):
            start_epoch = time.time()

            # Track time for each step within an epoch
            epoch_times = {
                "zero_grad": [],
                "forward_pass": [],
                "loss_calculation": [],
                "backward_pass": [],
                "optimizer_step": [],
                "db_insert": [],
                "batch": [],
                "test_forward": [],
                "test_loss_calc": [],
                "test_db_insert": [],
                "test_batch": [],
                "eval": [],
                "checkpoint": [],
                "scheduler": [],
            }

            self.model.train()
            total_train_loss = 0.0
            total_train_mse_loss = 0.0
            total_train_bce_loss= 0.0
            total_train_kl_loss = 0.0
            total_train_tc_loss = 0.0
            total_train_rn_loss = 0.0
            total_train_wd_loss = 0.0
            loss_data = {}
            for batch_idx, (x_batch, rn_percentage_batch) in enumerate(self.train_dataloader):
                start_batch = time.time()

                # Zero grad
                start_zero_grad = time.time()
                optimizer.zero_grad()
                end_zero_grad = time.time()
                epoch_times["zero_grad"].append(end_zero_grad - start_zero_grad)

                # Forward pass
                start_forward_pass = time.time()
                x_batch = x_batch.to(self.device)
                x_batch_masked = apply_random_masking_strategy(x_batch, mask_type_probabilities ={'contiguous': self.masking_strategy_contiguous_prob,
                                                                                                  'random': self.masking_strategy_random_prob},
                                                               contiguous_mask_length_ratio=self.contiguous_mask_length_ratio,
                                                               contiguous_min_num_masks=self.contiguous_min_num_masks,
                                                               contiguous_max_num_masks=self.contiguous_max_num_masks,
                                                               random_mask_ratio=self.random_mask_ratio)
                # original_data_point_for_plot = x_batch[0].cpu().numpy()
                # masked_data_point_for_plot = x_batch_masked[0].cpu().numpy()
                #
                # plt.plot(original_data_point_for_plot, label="original")
                # plt.plot(masked_data_point_for_plot, label="masked")
                # plt.legend()
                # plt.show()
                rn_percentage_batch = rn_percentage_batch.to(self.device)
                x_recon, rn_per, mu, logvar = self.model(x_batch_masked)
                end_forward_pass = time.time()
                epoch_times["forward_pass"].append(end_forward_pass - start_forward_pass)

                # Loss calculation
                start_loss_calculation = time.time()

                # loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                loss = torch.zeros(1, device=self.device, requires_grad=True)
                sample_size = x_batch.shape[0]

                temp_losses_arr = []
                if "mse" in self.losses:
                    mse_loss = mse_reconstruction_loss(x_recon, x_batch, reduction_mode='sum').to(self.device)
                    if self.is_loss_clip:
                        mse_loss = torch.clamp(mse_loss, max=self.loss_max).to(self.device)
                    loss = loss + mse_loss
                    temp_losses_arr.append(mse_loss)
                    total_train_mse_loss += mse_loss
                    # loss_data["mse"] = mse_loss.item() / sample_size
                if "bce" in self.losses:
                    bce_loss = bce_reconstruction_loss(x_recon, x_batch, reduction_mode='mean').to(self.device)
                    if self.is_loss_clip:
                        bce_loss = torch.clamp(bce_loss, max=self.loss_max).to(self.device)
                    loss = loss + bce_loss
                    temp_losses_arr.append(bce_loss)
                    total_train_bce_loss += bce_loss
                if "kl" in self.losses:
                    kl_loss = kl_divergence_loss(mu, logvar).to(self.device)
                    if self.is_loss_clip:
                        kl_loss = torch.clamp(kl_loss, max=self.loss_max).to(self.device)
                    loss = loss + kl_loss
                    temp_losses_arr.append(kl_loss)
                    total_train_kl_loss += kl_loss
                    # loss_data["kl"] = kl_loss.item() / sample_size
                if "wd" in self.losses:
                    wd_loss = wasserstein_distance_loss(x_recon, x_batch).to(self.device)
                    if self.is_loss_clip:
                        wd_loss = torch.clamp(wd_loss, max=self.loss_max).to(self.device)
                    loss = loss + wd_loss
                    temp_losses_arr.append(wd_loss)
                    total_train_wd_loss += wd_loss
                    # loss_data["wd"] = wd_loss.item() / sample_size
                if "tc" in self.losses:
                    tc_loss = total_counts_loss(x_recon, x_batch).to(self.device)
                    if self.is_loss_clip:
                        tc_loss = torch.clamp(tc_loss, max=self.loss_max).to(self.device)
                    loss = loss + tc_loss
                    temp_losses_arr.append(tc_loss)
                    total_train_tc_loss += tc_loss
                    # loss_data["tc"] = tc_loss.item() / sample_size
                if "rn" in self.losses:
                    rn_loss = mse_reconstruction_loss(rn_per, rn_percentage_batch).to(self.device)
                    if self.is_loss_clip:
                        rn_loss = torch.clamp(rn_loss, max=self.loss_max).to(self.device)
                    loss = loss + rn_loss
                    temp_losses_arr.append(rn_loss)
                    total_train_rn_loss += rn_loss
                    # loss_data["rn"] = rn_loss.item() / sample_size
                # loss_data["loss"] = loss.item() / sample_size
                # loss = torch.zeros(1, device=self.device, requires_grad=True)
                # loss = compute_metric(temp_losses_arr)
                end_loss_calculation = time.time()
                epoch_times["loss_calculation"].append(end_loss_calculation - start_loss_calculation)

                # Check if the loss is nan, otherwise exit the program.
                if torch.isnan(loss):
                    self.logger.error("Loss is NaN. Exiting execution...")
                    sys.exit()

                # Backward pass
                start_backward_pass = time.time()
                loss.backward()
                end_backward_pass = time.time()
                epoch_times["backward_pass"].append(end_backward_pass - start_backward_pass)

                # total_norm = 0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.detach().data.norm(2)
                #         total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** 0.5  # Compute L2 norm
                # self.logger.warning(f"Gradient norm: {total_norm:.4f}")
                # print(f"Gradient norm: {total_norm:.4f}")

                if self.is_gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_max)
                    # min_clip_value = 1e-5  # Set your minimum gradient threshold
                    # for param in self.model.parameters():
                    #     if param.grad is not None:
                    #         param.grad.data = torch.clamp(param.grad.data, min=min_clip_value)
                # Optimizer step
                start_optimizer_step = time.time()
                optimizer.step()
                end_optimizer_step = time.time()
                epoch_times["optimizer_step"].append(end_optimizer_step - start_optimizer_step)

                total_train_loss += loss

                

                # if batch_idx % self.training_batch_per_epoch_log_freq == 0:
                #     # Logging loss details
                #     # loss_parts = [f"Loss: {loss.item() / sample_size:.6f}"]
                #     if "mse" in self.losses:
                #         loss_parts.append(f"MSE Loss: {mse_loss.item() / sample_size:.6f}")
                #     if "kl" in self.losses:
                #         loss_parts.append(f"KL Loss: {kl_loss.item() / sample_size:.6f}")
                #     if "tc" in self.losses:
                #         loss_parts.append(f"TC Loss: {tc_loss.item() / sample_size:.6f}")
                #     if "wd" in self.losses:
                #         loss_parts.append(f"WD Loss: {wd_loss.item() / sample_size:.6f}")
                #     if "rn" in self.losses:
                #         loss_parts.append(f"RN Loss: {rn_loss.item() / sample_size:.6f}")
                #     info_msg = f"Epoch [{epoch + 1}/{self.epoch}], Batch [{batch_idx + 1}], {', '.join(loss_parts)}"
                #     self.logger.info(info_msg)

                end_batch = time.time()
                epoch_times["batch"].append(end_batch - start_batch)


            train_msg = f"Epoch [{epoch + 1}]:, Device: {self.device}, Training --> "
            avg_train_loss = total_train_loss.item() / self.train_dataset_size
            train_msg += f"Loss: {avg_train_loss:.6f}, "
            if "mse" in self.losses:
                loss_data["mse"] = total_train_mse_loss.item() / self.train_dataset_size
                train_msg += f"MSE: {loss_data['mse']:.6f}, "
            if "bce" in self.losses:
                loss_data["bce"] = total_train_bce_loss.item() / self.train_dataset_size
                train_msg += f"BCE: {loss_data['bce']:.6f}, "
            if "kl" in self.losses:
                loss_data["kl"] = total_train_kl_loss.item() / self.train_dataset_size
                train_msg += f"KL: {loss_data['kl']:.6f}, "
            if "wd" in self.losses:
                loss_data["wd"] = total_train_wd_loss.item() / self.train_dataset_size
                train_msg += f"WD: {loss_data['wd']:.6f}, "
            if "tc" in self.losses:
                loss_data["tc"] = total_train_tc_loss.item() / self.train_dataset_size
                train_msg += f"TC: {loss_data['tc']:.6f}, "
            if "rn" in self.losses:
                loss_data["rn"] = total_train_rn_loss.item() / self.train_dataset_size
                train_msg += f"RN: {loss_data['rn']:.6f}, "

            loss_data["loss"] = avg_train_loss

            self.logger.info(train_msg)


            train_loss_data_buffer.append(
                {
                    "db_table_name": self.train_loss_table_name,
                    "epoch": epoch +1,
                    "batch": batch_idx,
                    "sample_size": self.train_dataset_size,
                    "loss_values": loss_data
                }
            )

            if len(train_loss_data_buffer) == self.db_save_frequency:
                start_db_insert = time.time()
                for data in train_loss_data_buffer:
                    db_handler.insert_loss_data(data["db_table_name"], epoch=data["epoch"], batch=data["batch"], sample_size=data["sample_size"], loss_values=data["loss_values"])
                end_db_insert = time.time()
                epoch_times["db_insert"].append(end_db_insert - start_db_insert)

                self.logger.info(f"Database entries made for {len(train_loss_data_buffer)} loss values in the training database")
                train_loss_data_buffer = []

            # db_handler.insert_loss_data(self.train_loss_table_name, epoch=epoch+1, batch=batch_idx,
            #                             sample_size=self.train_dataset_size, loss_values=loss_data)

            
            self.model.eval()
            total_test_loss = 0.0
            total_test_mse_loss = 0.0
            total_test_bce_loss = 0.0
            total_test_kl_loss = 0.0
            total_test_tc_loss = 0.0
            total_test_rn_loss = 0.0
            total_test_wd_loss = 0.0
            loss_data = {}
            start_eval = time.time()
            with torch.no_grad():
                for batch_idx, (x_batch, rn_percentage_batch) in enumerate(self.test_dataloader):
                    start_test_batch = time.time()

                    # Test forward pass
                    start_test_forward = time.time()
                    x_batch = x_batch.to(self.device)
                    x_batch_masked = apply_random_masking_strategy(x_batch,  mask_type_probabilities ={'contiguous': self.masking_strategy_contiguous_prob,
                                                                                                  'random': self.masking_strategy_random_prob},
                                                               contiguous_mask_length_ratio=self.contiguous_mask_length_ratio,
                                                               contiguous_min_num_masks=self.contiguous_min_num_masks,
                                                               contiguous_max_num_masks=self.contiguous_max_num_masks,
                                                               random_mask_ratio=self.random_mask_ratio)
                    rn_percentage_batch = rn_percentage_batch.to(self.device)
                    x_recon, rn_per, mu, logvar = self.model(x_batch_masked)
                    end_test_forward = time.time()
                    epoch_times["test_forward"].append(end_test_forward - start_test_forward)

                    # Test loss calculation
                    start_test_loss_calc = time.time()
                    loss_data = {}
                    loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    sample_size = x_batch.shape[0]
                    temp_losses_arr = []
                    if "mse" in self.losses:
                        mse_loss = mse_reconstruction_loss(x_recon, x_batch, reduction_mode='sum').to(self.device)
                        if self.is_loss_clip:
                            mse_loss = torch.clamp(mse_loss, max=self.loss_max).to(self.device)
                        loss = loss + mse_loss
                        temp_losses_arr.append(mse_loss)
                        total_test_mse_loss += mse_loss
                    if "bce" in self.losses:
                        bce_loss = bce_reconstruction_loss(x_recon, x_batch, reduction_mode='mean').to(self.device)
                        if self.is_loss_clip:
                            bce_loss = torch.clamp(bce_loss, max=self.loss_max).to(self.device)
                        loss = loss + bce_loss
                        temp_losses_arr.append(bce_loss)
                        total_test_bce_loss += bce_loss
                    if "kl" in self.losses:
                        kl_loss = kl_divergence_loss(mu, logvar).to(self.device)
                        if self.is_loss_clip:
                            kl_loss = torch.clamp(kl_loss, max=self.loss_max).to(self.device)
                        loss = loss + kl_loss
                        temp_losses_arr.append(kl_loss)
                        total_test_kl_loss += kl_loss
                    if "wd" in self.losses:
                        wd_loss = wasserstein_distance_loss(x_recon, x_batch).to(self.device)
                        if self.is_loss_clip:
                            wd_loss = torch.clamp(wd_loss, max=self.loss_max).to(self.device)
                        loss = loss + wd_loss
                        temp_losses_arr.append(wd_loss)
                        total_test_wd_loss += wd_loss
                    if "tc" in self.losses:
                        tc_loss = total_counts_loss(x_recon, x_batch).to(self.device)
                        if self.is_loss_clip:
                            tc_loss = torch.clamp(tc_loss, max=self.loss_max).to(self.device)
                        loss = loss + tc_loss
                        temp_losses_arr.append(tc_loss)
                        total_test_tc_loss += tc_loss
                    if "rn" in self.losses:
                        rn_loss = mse_reconstruction_loss(rn_per, rn_percentage_batch).to(self.device)
                        if self.is_loss_clip:
                            rn_loss = torch.clamp(rn_loss, max=self.loss_max).to(self.device)
                        loss = loss + rn_loss
                        temp_losses_arr.append(rn_loss)
                        total_test_rn_loss += rn_loss
                    end_test_loss_calc = time.time()
                    epoch_times["test_loss_calc"].append(end_test_loss_calc - start_test_loss_calc)

                    # loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                    # loss = compute_metric(temp_losses_arr)


                    total_test_loss += loss

                    end_test_batch = time.time()
                    epoch_times["test_batch"].append(end_test_batch - start_test_batch)

                test_msg = f"Epoch [{epoch + 1}]:, Device: {self.device}, Testing --> "
                avg_test_loss = total_test_loss.item() / self.test_dataset_size
                test_msg += f"Loss: {avg_test_loss:.6f}, "

                if "mse" in self.losses:
                    loss_data["mse"] = total_test_mse_loss.item() / self.test_dataset_size
                    test_msg += f"MSE: {loss_data['mse']:.6f}, "
                if "bce" in self.losses:
                    loss_data["bce"] = total_test_bce_loss.item() / self.test_dataset_size
                    test_msg += f"BCE: {loss_data['bce']:.6f}, "
                if "kl" in self.losses:
                    loss_data["kl"] = total_test_kl_loss.item() / self.test_dataset_size
                    test_msg += f"KL: {loss_data['kl']:.6f}, "
                if "wd" in self.losses:
                    loss_data["wd"] = total_test_wd_loss.item() / self.test_dataset_size
                    test_msg += f"WD: {loss_data['wd']:.6f}, "
                if "tc" in self.losses:
                    loss_data["tc"] = total_test_tc_loss.item() / self.test_dataset_size
                    test_msg += f"TC: {loss_data['tc']:.6f}, "
                if "rn" in self.losses:
                    loss_data["rn"] = total_test_rn_loss.item() / self.test_dataset_size
                    test_msg += f"RN: {loss_data['rn']:.6f}, "

                loss_data["loss"] = avg_test_loss

                self.logger.info(test_msg)

                # Test DB insert
                # start_test_db_insert = time.time()
                # db_handler.insert_loss_data(self.test_loss_table_name, epoch=epoch, batch=batch_idx,
                #                             sample_size=self.test_dataset_size, loss_values=loss_data)
                # end_test_db_insert = time.time()
                # epoch_times["test_db_insert"].append(end_test_db_insert - start_test_db_insert)

                test_loss_data_buffer.append(
                    {
                        "db_table_name": self.test_loss_table_name,
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "sample_size": self.test_dataset_size,
                        "loss_values": loss_data
                    }
                )

                if len(test_loss_data_buffer) == self.db_save_frequency:
                    start_test_db_insert = time.time()
                    for data in test_loss_data_buffer:
                        db_handler.insert_loss_data(data["db_table_name"], epoch=data["epoch"], batch=data["batch"],
                                                    sample_size=data["sample_size"], loss_values=data["loss_values"])
                    end_test_db_insert = time.time()
                    epoch_times["test_db_insert"].append(end_test_db_insert - start_test_db_insert)

                    self.logger.info(
                        f"Database entries made for {len(test_loss_data_buffer)} loss values in the test database")
                    test_loss_data_buffer = []
                
                end_eval = time.time()
                epoch_times["eval"].append(end_eval - start_eval)

                # self.logger.info(
                #     f'''-->Epoch [{epoch + 1}/{self.epoch}], Average Train Loss: {avg_train_loss:.6f}, Average Test Loss: {avg_test_loss:.6f}'''
                # )

                if (epoch + 1) >= self.start_epoch_checkpoint and (epoch + 1) % self.checkpoint_freq == 0:
                    start_checkpoint = time.time()
                    model_name = f'''model_run_id_{self.run_id}_epoch_{epoch + 1}.pth'''
                    model_path = os.path.join(self.check_points, model_name)
                    torch.save(self.model.state_dict(), model_path)
                    end_checkpoint = time.time()
                    epoch_times["checkpoint"].append(end_checkpoint - start_checkpoint)

                if scheduler is not None:
                    start_scheduler_step = time.time()
                    scheduler.step()
                    end_scheduler_step = time.time()
                    epoch_times["scheduler"].append(end_scheduler_step - start_scheduler_step)

                end_epoch = time.time()
                epoch_duration = end_epoch - start_epoch

                avg_time_duration = []
                sum_time_list = []
                for step, times in epoch_times.items():
                    avg_time_duration.append(sum(times) / len(times) if times else 0)
                    sum_time_list.append(sum(times)) if times else 0

                total_av_time = sum(avg_time_duration)
                total_sum_time = sum(sum_time_list)

                # Log average times for each step
                # for step, times in epoch_times.items():
                #     avg_time = sum(times) / len(times) if times else 0
                #     sum_times = sum(times) if times else 0
                #     self.logger.info(f"Epoch [{epoch + 1}], Device: {self.device} Total {step} time: {sum_times:.4f} seconds. which took approximately: {((sum_times*100) / total_sum_time):.4f} %")

                self.logger.info(f"Epoch [{epoch + 1}] took: {epoch_duration:.4f} seconds.")
        if len(train_loss_data_buffer)>0:
            for data in train_loss_data_buffer:
                db_handler.insert_loss_data(data["db_table_name"], epoch=data["epoch"], batch=data["batch"],
                                            sample_size=data["sample_size"], loss_values=data["loss_values"])
            self.logger.info(
                f"Database entries made for {len(train_loss_data_buffer)} loss values in the training database")
            train_loss_data_buffer = []
        if len(test_loss_data_buffer) >= self.db_save_frequency:
            for data in test_loss_data_buffer:
                db_handler.insert_loss_data(data["db_table_name"], epoch=data["epoch"], batch=data["batch"],
                                            sample_size=data["sample_size"], loss_values=data["loss_values"])

            self.logger.info(
                f"Database entries made for {len(test_loss_data_buffer)} loss values in the test database")
            test_loss_data_buffer = []
