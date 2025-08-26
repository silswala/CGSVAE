import torch
import torch.nn as nn
import torch.nn.functional as F
from gudhi.wasserstein import wasserstein_distance

from itertools import combinations
import math

# Loss Functions


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """Computes the VAE loss, combining reconstruction loss and KL divergence.

    Args:
        x_recon (torch.Tensor): Reconstructed data from the decoder.
        x (torch.Tensor): Original input data.
        mu (torch.Tensor): Mean from the encoder's latent space.
        logvar (torch.Tensor): Log variance from the encoder's latent space.
        beta (float, optional): Hyperparameter to weight the KL divergence term. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the total VAE loss, reconstruction loss, and KL divergence.
               (total_loss, recon_loss, kl_divergence)
    """
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')  # MSE reconstruction loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
    total_loss = recon_loss + beta * kl_divergence  # Total VAE loss
    return total_loss, recon_loss, kl_divergence


def vae_loss_wd_for_recon_and_kl(x_recon, x, mu, logvar, beta=1.0):
    """Computes the VAE loss using Wasserstein distance for reconstruction and KL divergence.

    Args:
        x_recon (torch.Tensor): Reconstructed data from the decoder.
        x (torch.Tensor): Original input data.
        mu (torch.Tensor): Mean from the encoder's latent space.
        logvar (torch.Tensor): Log variance from the encoder's latent space.
        beta (float, optional): Hyperparameter to weight the KL divergence term. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the total VAE loss, reconstruction loss, and KL divergence.
               (total_loss, recon_loss, kl_divergence)
    """
    import numpy as np  # Import numpy inside the function where it is used.
    import gudhi  # Import gudhi inside the function where it is used.
    recon_nor = x_recon.detach().numpy() / (np.sum(x_recon.detach().numpy()))
    original_nor = x.detach().numpy() / (np.sum(x.detach().numpy()))
    recon_loss = gudhi.wasserstein.wasserstein_distance(recon_nor, original_nor, order=2)

    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_divergence
    return total_loss, recon_loss, kl_divergence


def mse_reconstruction_loss(x_recon, x, reduction_mode='sum', is_rn_loss=False):
    """Computes the Mean Squared Error (MSE) reconstruction loss.

    Args:
        x_recon (torch.Tensor): Reconstructed data from the decoder.
        x (torch.Tensor): Original input data.
        reduction_mode (str, optional): Reduction mode for the loss.
            Can be 'mean' or 'sum'. Defaults to 'sum'.
        is_rn_loss (bool, optional): A flag to indicate if this is a
            "reconstruction normalized" loss (Radionuclide Percentage Loss).
            If True, the loss is multiplied by 10000. This scaling is
            applied because the predicted values (`x_recon`) are between 0
            and 1, and multiplying by 10000 converts the loss to a percentage
            scale. Defaults to False.

    Returns:
        torch.Tensor: The MSE reconstruction loss. If `is_rn_loss` is True,
                      the returned loss will be scaled by 10000.
    """
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction=reduction_mode)
    if is_rn_loss:
        recon_loss = recon_loss * 10000
    return recon_loss


def bce_reconstruction_loss(x_recon, x, reduction_mode='sum'):
    """Computes the Binary Cross-Entropy (BCE) reconstruction loss.

	Args:
	    x_recon (torch.Tensor): Reconstructed data from the decoder, expected to be probabilities (between 0 and 1).
	    x (torch.Tensor): Original input data, expected to be binary (0 or 1).
	    reduction_mode (str, optional): Specifies the reduction to apply to the output.
	        Can be 'none', 'mean', or 'sum'. Defaults to 'sum'.

	Returns:
	    torch.Tensor: The BCE reconstruction loss.
	"""
    return nn.BCELoss(reduction=reduction_mode)(x_recon, x)


def kl_divergence_loss(mu, logvar):
    """Computes the KL divergence loss.

    Args:
        mu (torch.Tensor): Mean of the latent variable distribution.
        logvar (torch.Tensor): Log variance of the latent variable distribution.

    Returns:
        torch.Tensor: The KL divergence loss.
    """
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


def wasserstein_distance_loss(x_recon, x):
    """Computes the Wasserstein distance loss.

	This function calculates the Wasserstein-2 distance (also known as the Earth Mover's Distance)
	between the distributions of the original and reconstructed data.

	Args:
	    x_recon (torch.Tensor): Reconstructed data from the decoder.
	    x (torch.Tensor): Original input data.

	Returns:
	    torch.Tensor: The Wasserstein distance loss as a PyTorch tensor.
	"""
    recon_nor = x_recon.cpu().detach().numpy()
    original_nor = x.detach().cpu().detach().numpy()

    # wd_loss = gudhi.wasserstein.wasserstein_distance(recon_nor, original_nor, order=2)
    wd_loss = wasserstein_distance(recon_nor, original_nor, order=2)
    wd_loss = torch.tensor(wd_loss)
    return wd_loss


def total_counts_loss(x_recon, x, reduction_mode='mean', validator :bool = False):
    """Computes the total counts loss, which penalizes differences in total
	summed values between original and reconstructed tensors.

	This loss is calculated as the squared difference between the total counts
	of the original and reconstructed tensors.

	Args:
	    x_recon (torch.Tensor): Reconstructed tensor.
	    x (torch.Tensor): Original tensor.
	    reduction_mode (str, optional): Specifies the reduction to apply to the output.
	        Can be 'mean' or 'sum'. Defaults to 'mean'.
	    validator (bool, optional): If True, returns the loss along with the
	        original and reconstructed total counts for validation purposes.
	        Defaults to False.

	Returns:
	    torch.Tensor or tuple: The total counts loss. If `validator` is True,
	                           returns a tuple `(loss, original_counts, reconstructed_counts)`.
	"""
    original_counts_total = torch.sum(x, dim=1)
    reconstructed_counts_total = torch.sum(x_recon, dim=1)

    if reduction_mode == 'mean':
        count_loss = torch.mean((original_counts_total - reconstructed_counts_total) ** 2)
    elif reduction_mode == 'sum':
        count_loss = torch.sum((original_counts_total - reconstructed_counts_total) ** 2)
    else:
        raise ValueError("Invalid reduction mode. Choose 'mean' or 'sum'.")  # Raise an error for invalid input

    if validator:
        return count_loss, original_counts_total, reconstructed_counts_total
    return count_loss


def poisson_fidelity_loss(x_recon,
                          x,
                          multiplier: int = 10000,
                          final_output_min_clamp = 1e-8,
                          final_output_max_clamp = 10000000.0):
    """Computes a Poisson fidelity loss based on the total counts of the
	original and reconstructed tensors.

	This function models the total counts as Poisson-distributed variables and
	calculates the Poisson Probability Mass Function (PMF). The PMF is used as a
	fidelity measure, with higher values indicating a better match. This function
	returns the PMF itself, not the negative log-likelihood typically used as a loss.

	Args:
	    x_recon (torch.Tensor): Reconstructed data tensor.
	    x (torch.Tensor): Original data tensor.
	    multiplier (int, optional): A scaling factor applied to the total counts
	        before computing the Poisson PMF. This is crucial for handling
	        floating-point inputs and converting them to integer-like counts.
	        Defaults to 10000.
	    final_output_min_clamp (float, optional): The minimum value to clamp the
	        final probability output to, preventing log-of-zero issues in subsequent
	        calculations if this is used within a larger loss function. Defaults to 1e-8.
	    final_output_max_clamp (float, optional): The maximum value to clamp the
	        final probability output to. Defaults to 10000000.0.

	Returns:
	    torch.Tensor: A scalar tensor representing the Poisson PMF of the observed
	        counts (`x_recon`) given the mean of the original counts (`x`).
	"""
    device = x.device
	# Define a small epsilon for log(0) and to ensure P is never exactly zero before final clamp
    epsilon = 1e-10

    # Global sum across all dimensions for x and x_recon, then multiply, round, and clamp
    poisson_mean = (torch.sum(x) * multiplier).round().clamp(min=0.0)
    poisson_observed = (torch.sum(x_recon) * multiplier).round().clamp(min=0.0)

    # Calculate log Poisson PMF for this single pair of values
    log_prob_poisson_scalar = (
        poisson_observed * torch.log(poisson_mean + epsilon) # k * log(lambda), add epsilon to mean for log(0)
        - poisson_mean # - lambda
        - torch.lgamma(poisson_observed + 1) # - log(k!) = - log_gamma(k+1)
    )

    # Special handling for mean=0
    if poisson_mean == 0:
        if poisson_observed == 0:
            log_prob_poisson_scalar = torch.tensor(0.0, device=device) # log(1) = 0
        else:
            log_prob_poisson_scalar = torch.tensor(-torch.inf, device=device) # log(0) = -inf

    # Convert log probability back to actual probability
    combined_poisson_probability_scalar = torch.exp(log_prob_poisson_scalar)

    # Add a small epsilon to the probability *before* the final clamp,
    # to ensure it's slightly above zero if it was exactly zero from exp(-inf),
    # making it adhere to the 1e-8 min clamp.
    combined_poisson_probability_scalar = combined_poisson_probability_scalar + epsilon

    # --- Clamp the final output probability to the specified range ---
    combined_poisson_probability_scalar = torch.clamp(
        combined_poisson_probability_scalar,
        min=final_output_min_clamp,
        max=final_output_max_clamp
    )

    return combined_poisson_probability_scalar


def rn_loss_l1(x_recon, x):
    """Computes the L1 loss (mean absolute error).

    Args:
        x_recon (torch.Tensor): Reconstructed tensor.
        x (torch.Tensor): Original tensor.

    Returns:
        torch.Tensor: The L1 loss.
    """
    return torch.mean(torch.abs(x - x_recon))


def cross_entropy_loss(predictions, targets):
    """Computes the Cross-Entropy Loss.

    Args:
        predictions (torch.Tensor): Predicted logits. Shape: [batch_size, num_classes]
        targets (torch.Tensor): True class indices. Shape: [batch_size]

    Returns:
        torch.Tensor: The Cross-Entropy Loss.
    """
    loss = F.cross_entropy(predictions, targets, reduction='mean')
    return loss


def compute_metric(values, method='sum_product'):
    """Computes a custom metric based on combinations and products of tensor values.

	This function calculates the sum of all possible products of combinations
	of the input tensors. For a list of tensors [v1, v2, v3], it computes:
	(v1) + (v2) + (v3) + (v1*v2) + (v1*v3) + (v2*v3) + (v1*v2*v3).

	Args:
	    values (list of torch.Tensor): A list of 1D tensors to compute the metric from.
	    method (str, optional): The method for computation. Currently, only
	        'sum_product' is supported. Defaults to 'sum_product'.

	Returns:
	    torch.Tensor: The computed metric value as a scalar tensor.

	Raises:
	    ValueError: If an unsupported `method` is provided or if the elements
	                in `values` are not PyTorch tensors.
	"""
    if method != 'sum_product':
        raise ValueError("Only 'sum_product' method is supported.")

    # Check if each element in the list is a PyTorch tensor
    if not all(isinstance(v, torch.Tensor) for v in values):
        raise ValueError("Each element in 'values' must be a PyTorch tensor.")

    device = values[0].device  # Get the device of the first element (assuming all elements are on the same device)
    n = len(values)

    # Initialize result as a tensor of type float on the correct device
    result = torch.tensor(0.0, device=device, dtype=torch.float)  # Explicitly set dtype to float

    # For k = 1 to n, sum of products of all combinations of size k
    for k in range(1, n + 1):
        for combo in combinations(values, k):
            # Ensure the product is computed on the same device as the elements
            product = torch.prod(torch.stack(combo, dim=0).to(torch.float))  # Ensure product is float
            result += product  # Accumulate the product on the same device

    return result
