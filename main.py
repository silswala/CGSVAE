import torch
import random
import numpy as np
import argparse

from sympy.physics.units import joule

from Utility import mylogging
from models.conditional_cgsvae import ConditionalCGSVAE
from models.cgsvae import CGSVAE
import json
from Utility.mylogging import *
from trainer import Trainer
from trainer_inpainting import InpaintingTrainer
import uuid
import datetime
from Utility.config_handler import validate_config, parse_config_json
from trainer_conditional import ConditionalTrainer


def seed_torch(seed: int = 3007):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    This function ensures deterministic behavior in model training and evaluation by initializing
    the random seed in the Python `random`, `numpy`, and `torch` libraries. It also sets PyTorch
    CUDA seeds and backend flags for consistent results in multi-GPU setups.

    Args:
        seed (int, optional): The random seed to use. Defaults to 3007.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.benchmark = False  # For consistent performance
    torch.backends.cudnn.deterministic = True  # For reproducible results


def parse_arguments():
    """
	Parses command-line arguments for a training experiment.

	This function sets up an `ArgumentParser` to define and parse various command-line
	options necessary for configuring a training run. It includes options for specifying
	config files, model type, compute device, logging directories, random seeds, and
	user-defined metadata.

	Returns:
		argparse.Namespace: An object containing the parsed command-line arguments as
							attributes.
	"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to the JSON config file.")
    parser.add_argument("--config_schema_file", default="./configs/config_schema.json", help="Path to the path of config schema.")
    parser.add_argument("--model_type", required=True, choices=["CGSVAE", "cCGSVAE", "InCGSVAE"], help="Type of model to use.")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use (cuda or cpu). Auto-selects if not provided.")
    parser.add_argument("--logdir", default="output", type=str, help="Directory to save logs and outputs.")
    parser.add_argument("--seed", type=int, help="Random seed for weight initialization.")
    parser.add_argument("--user_run_id", default="", type=str, help="Some string that user thinks will identify run uniquely")
    parser.add_argument("--user_comment", default="", type=str,
                        help="Any other information about the specific run")

    return parser.parse_args()


def generate_run_id()  -> str:
    """
    Generates a unique identifier for a training run.

    This function combines the current timestamp with a short, unique UUID to create a string
    that can be used as a run ID for logging or directory naming.

    Returns:
        str: A unique run ID string in the format "YYYYMMDD_HHMMSS_XXXXXXXX", where
             "XXXXXXXX" is an 8-character part of a UUID.
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_part = str(uuid.uuid4()).replace("-", "")[:8]
    return f"{now}_{unique_part}"


def main():
    """
    Main function to run the training experiment.

    This function orchestrates the entire training pipeline for a VAE-based model. It performs the following steps:
    1.  Argument Parsing: Calls `parse_arguments()` to get command-line arguments.
    2.  Configuration: Loads the base configuration from a JSON file, generates a unique run ID, and creates the necessary output directories (run directory, checkpoints). It then updates the configuration dictionary with run-specific paths and information.
    3.  Logging: Sets up a logger to save all informational, warning, and error messages to a log file within the run's directory.
    4.  Device and Seed Setup: Determines the appropriate device (`cuda` or `cpu`) based on arguments and availability, and sets a random seed for reproducibility.
    5.  Configuration Validation: Validates the loaded configuration against a predefined schema to ensure all necessary parameters are present and correctly formatted.
    6.  Model Initialization: Instantiates the correct model (`CGSVAE`, `InCGSVAE`, or `cCGSVAE`) based on the `model_type` specified in the arguments.
    7.  Training: Initializes a `Trainer` object and starts the training process for the selected model.
    
    The function handles all setup and execution, but does not return any value.
    """
    
    #
    # Argument Parsing
    #
    args = parse_arguments()

    # 
    # Configuration
    # 
    config = parse_config_json(args.config_file)
    config["run_info"] = {}
    run_info = config["run_info"]

    run_id = generate_run_id()
    run_info["run_id"] = run_id
    run_id = f"{run_id}{args.user_run_id}"
    run_dir = os.path.join(args.logdir,  run_id)
    os.makedirs(run_dir, exist_ok=True)
    run_info["run_dir"] = run_dir

    checkpoints_dir = os.path.join(run_dir, f"checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    run_info["check_points"] = checkpoints_dir

    log_file_name = os.path.join(run_dir, f"model_run_{run_id}.log")
    run_info["log_file_name"] = log_file_name
    config_file_name = os.path.join(run_dir, f"config_{run_id}.json")
    run_info["config_file_name"] = config_file_name

    config_schema_file_name = os.path.join(run_dir, f"config_schema{run_id}.json")
    run_info["config_schema_name"] = config_schema_file_name

    loss_file_name = os.path.join(run_dir, f"output_{run_id}.db")
    run_info["loss_file_name"] = loss_file_name

    _log_file_path = log_file_name
    
    
    
    #
    # Logging
    #
    logger = mylogging.setup_logging(log_file_path=_log_file_path, logger_object_name="main")
    
    

    #
    # Device and Seed Setup
    #
    if args.device:
        device = args.device
        logger.info(f"Using device specified in arguments: {device}")
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.error("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
                logger.info("Using CPU.")
            else:
                device_count = torch.cuda.device_count()
                logger.info(f"Number of CUDA devices available: {device_count}")

    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available. Using CUDA.")
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices available: {device_count}")

    else:
        device = "cpu"
        logger.info("CUDA not available. Using CPU.")
    config["device"] = device

    _seed = int(args.seed) if args.seed is not None else 42  # If seed is not provided, seed is set to 42.
    seed_torch(seed=_seed)
    logger.info(f"Seed: {_seed}")
    config["seed"] = _seed
    config["model"]["name"] = args.model_type


    if args.user_comment:
        config["user_comment"] = args.user_comment

    # Save updated config
    with open(config_file_name, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Updated config saved to: {config_file_name}")

    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Log directory: {args.logdir}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Run ID: {run_id}")
    
    
    

    #
    # Configuration Validation
    #
    config_schema = parse_config_json(args.config_schema_file)
    with open(config_schema_file_name, "w") as f:
        json.dump(config_schema, f, indent=4)
    validate_config(config, config_schema, strict=True)
    
    logger.info(f"Config: {config}")
    
    
    
    #
    # Model Initialization and Training
    #
    if args.model_type == "CGSVAE":
        #
        # Model Definition
        #
        logger.info("Model name: {model}".format(model=args.model_type))
        model = CGSVAE(config=config, initialize_weights=True)
        logger.info("Model summary: {model}".format(model=model))
        logger.info(horizontal_line(length=50, space=0, char="--"))
        logger.info("Model training is starting....")
        logger.info(horizontal_line(length=50, space=0, char="--"))

        #
        # Training Model
        #
        trainer = Trainer(model=model, config=config)
        trainer.train_model()

    if args.model_type == "InCGSVAE":
        #
        # Model Definition
        #
        logger.info("Model name: {model}".format(model=args.model_type))
        model = CGSVAE(config=config, initialize_weights=True)
        logger.info("Model summary: {model}".format(model=model))
        logger.info(horizontal_line(length=50, space=0, char="--"))
        logger.info("Model training is starting....")
        logger.info(horizontal_line(length=50, space=0, char="--"))

        #
        # Training Model
        #
        trainer = InpaintingTrainer(model=model, config=config)
        trainer.train_model()

    if args.model_type == "cCGSVAE":
        #
        # Model Definition
        #
        logger.info("Model name: {model}".format(model=args.model_type))
        model = ConditionalCGSVAE(config=config, initialize_weights=True)
        logger.info("Model summary: {model}".format(model=model))
        logger.info(horizontal_line(length=50, space=0, char="--"))
        logger.info("Model training is starting....")
        logger.info(horizontal_line(length=50, space=0, char="--"))

        #
        # Training Model
        #
        trainer = ConditionalTrainer(model=model, config=config)
        trainer.train_model()



if __name__ == '__main__':
    main()
