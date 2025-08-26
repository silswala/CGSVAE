import json
import os

def create_json_config(filepath, config_data):
    """
    Creates a JSON config file at the specified location, ensuring that the
    parent directory exists. If the directory does not exist, it will be created.

    Args:
        filepath (str): Path to the output JSON file.
        config_data (dict): Dictionary containing the configuration data.

    Returns:
        None
    """
    # Ensure the parent directory exists
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(config_data, f, indent=4)


def generate_schema(config_data):
    """Generates a schema dictionary from config_data."""

    def get_type(value):
        """Returns the type of a value in a JSON-compatible format."""
        if isinstance(value, list):
            return ["list", get_type(value[0]) if value else "unknown"]  # Handle empty lists
        return type(value).__name__

    schema = {}
    for key, value in config_data.items():
        if isinstance(value, dict):
            schema[key] = generate_schema(value)  # Recursively process dictionaries
        else:
            schema[key] = get_type(value)

    return schema

def create_json_schema(filepath, schema_data):
    """Creates a JSON schema file.

    Args:
        filepath (str): Path to the output JSON schema file (without extension).
        schema_data (dict): Dictionary containing the schema.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists

    with open(filepath + ".json", "w") as f:
        json.dump(schema_data, f, indent=4)



config_data = {
    "model": {  # Model-related parameters
        "input_dim": 1024,
        "encoder_hidden_dims": [512, 256],
        "latent_dim": 8,
        "decoder_hidden_dims": [256, 512],
        "output_dim": 1024,
        "condition_dim": 6,
        "additional_output_dim": 6,  # If applicable, add more model params
        "weight_initialization_method": "kaiming_uniform",  # Other options are "xavier_normal", "xavier_uniform, "kaiming_uniform, kaiming_normal"
        "rn_activation": "Softmax"    # Only if Softmax is given, Other options are "Sigmoid"
    },
    "training":
        {  # Training-related parameters
            "learning_rate": 0.001,
            "optimizer": "adam",  # name of optimizer, accordingly implement to handle in sepecvaeutil.py file
            "epoch": 10,
            "batch_size": 2048,
            "shuffle_train_data": True,
            "shuffle_test_data": False,
            "shuffle_val_data": True,  # If you use validation data
            "losses": ["mse", "kl", "rn"],  # Options are "mse", "kl", "tc", "rn", "wd", "bce", "pf"
            "scheduler": {  # Learning rate scheduler parameters (nested)
                "name": "steplr",  # "steplr"
                "step_size": 100,
                "gamma": 0.9
            },
            "logging":
            {
                "training_batch_per_epoch_log_freq": 500,
                "checkpoint_freq": 1,
                "start_epoch_checkpoint": 10
            }
        },
    "data": {  # Data-related parameters
        "train_db_names": [
            r"/Users/amitsilswal/Downloads/CGSVAE/SampleData/train.db",
        ],  # List of file locations of train data
        "test_db_names": [
             r"/Users/amitsilswal/Downloads/CGSVAE/SampleData/test.db",
        ],  # List of file locations of test data
        "val_db_names": [
             r"/Users/amitsilswal/Downloads/CGSVAE/SampleData/validation.db",
        ]  # List of file locations of validation data (if used)
    },
    "masking":
        {
            "masking_strategy":
                {
                    "contiguous": 0.7,
                    "random": 0.3
                },
                "contiguous_mask_length_ratio": 0.1,
                "contiguous_min_num_masks": 1,
                "contiguous_max_num_masks": 5,
                "random_mask_ratio": 0.5
        },
    "train_loss_table_name": "train_losses",
    "test_loss_table_name": "test_losses",
    "user_comment": "",
    "db_save_frequency": 10
}

schema_data = {
    "model": {
        "name": "",
        "input_dim": 0,
        "encoder_hidden_dims": [0],
        "latent_dim": 0,
        "decoder_hidden_dims": [0],
        "output_dim": 0,
        "condition_dim": 0,
        "additional_output_dim": 0,
        "weight_initialization_method": "",
        "rn_activation": ""
    },
    "training":
        {
            "learning_rate": 0.0,
            "optimizer": "",
            "epoch": 0,
            "batch_size": 0,
            "shuffle_train_data": False,
            "shuffle_test_data": False,
            "shuffle_val_data": False,
            "losses": [""],
            "scheduler": {
                "name": "",
                "step_size": 0,
                "gamma": 0.0
            },
            "logging":
                {
                    "training_batch_per_epoch_log_freq": 0,
                    "checkpoint_freq": 0,
                    "start_epoch_checkpoint": 0
                }
            ,
        },
    "data": {
        "train_db_names": [""],
        "test_db_names": [""],
        "val_db_names": [""],
    },"masking":
        {
            "masking_strategy":
                {
                    "contiguous": 0.0,
                    "random": 0.0
                },
                "contiguous_mask_length_ratio": 0.0,
                "contiguous_min_num_masks": 0,
                "contiguous_max_num_masks": 0,
                "random_mask_ratio": 0.0
        }
    ,
    "run_info": {
        "run_id": "",
        "run_dir": "",
        "check_points": "",
        "log_file_name": "",
        "config_file_name": "",
        "config_schema_name": "",
        "loss_file_name": ""
    },
    "device": "",
    "seed": 0,
    "train_loss_table_name": "",
    "test_loss_table_name": "",
    "user_comment": "",
    "db_save_frequency": 0
}

create_json_config(f"configs/config", config_data)
create_json_schema("configs/config_schema", generate_schema(schema_data))
