import sys

import torch
import torch.nn as nn
from torch.nn import Softmax

from Utility.config_handler import get_nested_config
from Utility.mylogging import setup_logging
from Utility.util import check_type


class Encoder(nn.Module):
    """The Encoder part of the Variational Autoencoder (VAE).

    Encodes input data into the parameters (mu, logvar) of a latent Gaussian distribution,
    which are then used to sample a latent vector `z` via the reparameterization trick.

    Args:
        config (dict): A dictionary containing configuration parameters, including:
            - input_dim (int): Dimension of the input data.
            - encoder_hidden_dims (list): List of hidden layer dimensions for the encoder.
            - latent_dim (int): Dimension of the latent space.

    Attributes:
        hidden_layers (nn.ModuleList): List of linear layers for the encoder's hidden layers.
        mu_layer (nn.Linear): Linear layer to output the mean (mu) of the latent distribution.
        logvar_layer (nn.Linear): Linear layer to output the log variance (logvar) of the latent distribution.

    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.input_dim = get_nested_config(config, ["model", "input_dim"])
        check_type(self.input_dim, int)
        self.hidden_dims = get_nested_config(config, ["model", "encoder_hidden_dims"])
        check_type(self.hidden_dims, list, element_type=int)
        self.latent_dim = get_nested_config(config, ["model", "latent_dim"])

        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.mu_layer = nn.Linear(prev_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, self.latent_dim)

    def forward(self, x):
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the mean (mu) and log variance (logvar) of the latent distribution.
        """
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar


class Decoder(nn.Module):
    """The Decoder part of the Variational Autoencoder (VAE).

    Decodes a latent vector (z) into reconstructed data (x_recon) and an additional output (rn_percent) if configured.

    Args:
        config (dict): A dictionary containing configuration parameters, including:
            - decoder_hidden_dims (list): List of hidden layer dimensions for the decoder.
            - latent_dim (int): Dimension of the latent space.
            - output_dim (int): Dimension of the reconstructed output.
            - additional_output_dim (int, optional): Dimension of the additional output. Defaults to 0.

    Attributes:
        hidden_layers (nn.ModuleList): List of linear layers for the decoder's hidden layers.
        output_layer (nn.Linear): Linear layer to output the reconstructed data (x_recon).
        rn_output_layer (nn.Linear, optional): Linear layer to output the additional output (rn_percent). Only present if additional_output_dim > 0.
        activation (nn.Sigmoid): Sigmoid activation function applied to the reconstructed output.

    """

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.logdir = get_nested_config(config=config, keys=["run_info", "log_file_name"])
        self.logger = setup_logging(log_file_path=self.logdir, logger_object_name="decoder")
        self.hidden_layers = nn.ModuleList()
        self.hidden_dims = get_nested_config(config, ["model", "decoder_hidden_dims"])
        self.latent_dim = get_nested_config(config, ["model", "latent_dim"])
        self.output_dim = get_nested_config(config, ["model", "output_dim"])
        self.additional_output_dim = get_nested_config(config, ["model", "additional_output_dim"])
        losses = get_nested_config(config, ["training", "losses"])
        # self.is_rn_loss_requested = "rn" in losses if losses else False

        prev_dim = self.latent_dim
        for h_dim in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        # if self.additional_output_dim > 0 and self.is_rn_loss_requested:
        if self.additional_output_dim > 0:
            self.rn_output_layer = nn.Linear(prev_dim, self.additional_output_dim)
        else:
            self.logger.warning("'additional_output_dim' is given in the config without 'rn' loss requested. Not adding any additional output dim in the model")
        #
        # Check here, we want radionuclide weights sum to one. So if we define in config: rn_activation as 'Softmax' than user softmax, otherwise always user Sigmoid
        #
        self.activation = nn.Sigmoid()
        rn_act = get_nested_config(config=config, keys=["model", "rn_activation"], default_value="Sigmoid")
        if rn_act and rn_act == "Softmax":
            self.rn_activation = nn.Softmax(dim=-1)
        else:
            self.rn_activation = nn.Sigmoid()


    def forward(self, z):
        """Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            tuple: A tuple containing the reconstructed data (x_recon) and the additional output (rn_percent).
                `rn_percent` is a tensor if `additional_output_dim > 0`, otherwise it is `None`.
        """
        x = z
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        x_recon = self.output_layer(x)
        x_recon = self.activation(x_recon)

        rn_percent = None  # Initialize to None
        # if self.additional_output_dim > 0 and self.is_rn_loss_requested:
        if self.additional_output_dim > 0:
            rn_percent = self.rn_output_layer(x)
            rn_percent = self.rn_activation(rn_percent)
        return x_recon, rn_percent


def _reparametrize(mu, logvar):
    """Reparameterization trick for sampling from the latent distribution.

    Samples from the latent distribution q(z|x) using the provided mu and logvar.
    This allows for gradient backpropagation through the sampling process.

    Args:
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log variance of the latent distribution.

    Returns:
        torch.Tensor: A sample from the latent distribution.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class CGSVAE(nn.Module):
    def __init__(self, config, initialize_weights: bool = False):
        """Initializes the CGSVAE model.

        Args:
            config (dict): A dictionary containing configuration parameters for the encoder and decoder, including the training device.
            initialize_weights (bool, optional): Whether to initialize the model's weights. Defaults to False.
        """
        super(CGSVAE, self).__init__()
        self.device = get_nested_config(config, ["device"])
        self.weight_initialization_method = get_nested_config(config, ["model", "weight_initialization_method"])
        self.logdir = get_nested_config(config=config, keys=["run_info", "log_file_name"])
        self.logger = setup_logging(log_file_path=self.logdir, logger_object_name="CGSVAE")
        self.config = config
        self.encoder = Encoder(self.config).to(self.device)
        self.decoder = Decoder(self.config).to(self.device)
        if initialize_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using the specified method from config. Logs errors if an invalid method is provided."""
        valid_methods = {"xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform"}

        if self.weight_initialization_method not in valid_methods:
            error_message = (f"Invalid weight initialization method: '{self.weight_initialization_method}'. "
                             f"Valid options are: {valid_methods}")
            self.logger.error(error_message)
            raise AssertionError(error_message)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_initialization_method == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                    self.logger.info("Initialized weights using Xavier Normal.")
                elif self.weight_initialization_method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                    self.logger.info("Initialized weights using Xavier Uniform.")
                elif self.weight_initialization_method == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    self.logger.info("Initialized weights using Kaiming Normal.")
                elif self.weight_initialization_method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                    self.logger.info("Initialized weights using Kaiming Uniform.")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    self.logger.info("Bias initialized to zero.")

    def forward(self, x):
        """Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the reconstructed data (x_recon), the additional output (rn_percent),
                the mean (mu) and log variance (logvar) of the latent distribution.
        """
        x = x.to(self.device)
        mu, logvar = self.encoder(x)
        
        
        # Clamp logvar to prevent numerical instability and exploding gradients during training.
        # The VAE's KL divergence loss term is highly sensitive to the values of logvar.
        #
        # Reason for min=-7 clamp: Prevents the standard deviation from becoming extremely small.
        # A tiny standard deviation (e.g., if logvar is -100) would make the latent distribution
        # a sharp spike. The KL loss would then 'explode' as it tries to regularize this
        # spike towards a standard normal distribution, leading to huge gradients.
        #
        # Reason for max=10 clamp: Prevents the standard deviation from becoming extremely large.
        # A very large standard deviation (e.g., if logvar is 10) would make the latent distribution
        # too broad. The KL loss would also generate large gradients to pull this wide distribution
        # back toward a standard normal, causing training to become unstable.
        logvar = torch.clamp(logvar, min=-7, max=10)     
        z = _reparametrize(mu, logvar)
        x_recon, rn_percent = self.decoder(z)
        return x_recon, rn_percent, mu, logvar
