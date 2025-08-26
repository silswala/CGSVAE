import logging
from Utility.mylogging import *
import json

class ConfigError(Exception):
    pass


def get_nested_config(config: dict, keys: list[str], default_value=None):
	"""Retrieves a nested value from a dictionary using a list of keys.

	This function safely navigates a nested dictionary structure. If a key is
	missing along the path, it can either return a specified default value
	or raise a `ConfigError`, depending on the provided `default_value`.

	Args:
	    config (dict): The dictionary from which to retrieve the value.
	    keys (list[str]): A list of strings representing the sequential keys
	        to traverse the nested dictionary.
	    default_value: The value to return if the final key is not found. If
	        `None` (default), a `ConfigError` is raised instead. Defaults to None.

	Returns:
	    The value found at the specified nested path.

	Raises:
	    ConfigError:
	        - If the required 'run_info' or 'log_file_name' keys are missing from
	          the top-level configuration.
	        - If any key along the path is missing and no `default_value` is provided
	          for the final key.
	        - If a value along the path is not a dictionary when a key traversal is expected.
	"""

	if "run_info" not in config or "log_file_name" not in config.get("run_info", {}):
		raise ConfigError("Essential keys 'run_info' and/or 'log_file_name' are missing from the configuration.")

	log_file_path = config["run_info"]["log_file_name"]
	logger = setup_logging(log_file_path=log_file_path, logger_object_name="config_handler")

	current = config
	for i, key in enumerate(keys):
		if isinstance(current, dict) and key in current:
			current = current[key]
		elif isinstance(current, dict):
			# Key is missing, check if a default value is provided
			if default_value is not None and i == len(keys) - 1: # Only apply default if it's the very last key that's missing
				section_name = ".".join(keys[:i]) if i > 0 else "configuration"
				warning_msg = f"Warning: Key '{key}' not found in '{section_name}'. Using default value: {default_value}"
				logger.warning(warning_msg) # Log as a warning
				return default_value
			else:
				section_name = ".".join(keys[:i]) if i > 0 else "configuration"
				error_msg = f"KeyError: Missing key '{key}' in '{section_name}'."
				logger.error(error_msg)
				raise ConfigError(error_msg)
		else:
			section_name = ".".join(keys[:i]) if i > 0 else "configuration"
			error_msg = f"TypeError: Expected a dictionary at '{section_name}', but got {type(current)}."
			logger.error(error_msg)
			raise ConfigError(error_msg)
	return current


def validate_config(config_data, schema_data, path="", strict=False):
    """Recursively validates a configuration dictionary against a schema.

	The function checks for type correctness and key presence based on the schema.
	It supports nested dictionaries and lists with specified element types.

	Args:
	    config_data (dict): The configuration data to be validated.
	    schema_data (dict): The schema dictionary defining the expected structure
	        and data types. Data types should be provided as strings (e.g., 'str', 'int').
	    path (str, optional): An internal parameter used for building the full
	        key path for error messages. Defaults to "".
	    strict (bool, optional): If `True`, the function raises an `AssertionError`
	        if `config_data` contains any key not present in `schema_data`.
	        Defaults to False.

	Raises:
	    AssertionError: If a type mismatch is found, or if a key is present in
	                    `config_data` but not in `schema_data` when `strict` is True.
	"""
    for key, value in config_data.items():
        full_path = f"{path}.{key}" if path else key  # Create a readable path

        if key not in schema_data and strict:
            raise AssertionError(f"Key '{full_path}' is not defined in the schema.")

        if key in schema_data:
            expected_type = schema_data[key]

            if isinstance(expected_type, dict):
                # If it's a nested dictionary, recurse
                assert isinstance(value, dict), f"{full_path} should be a dict, got {type(value).__name__}"
                validate_config(value, expected_type, full_path, strict)

            elif isinstance(expected_type, list):
                # If it's a list, check the type of elements
                assert isinstance(value, list), f"{full_path} should be a list, got {type(value).__name__}"

                if value:  # Only check element types if the list is not empty
                    expected_element_type = expected_type[1]
                    for idx, item in enumerate(value):
                        assert isinstance(item, eval(expected_element_type)), (
                            f"{full_path}[{idx}] should be {expected_element_type}, got {type(item).__name__}"
                        )

            else:
                assert isinstance(value, eval(expected_type)), (
                    f"{full_path} should be {expected_type}, got {type(value).__name__}"
                )

def parse_config_json(configfile: str) -> dict:
    """Parses and loads a JSON configuration file into a dictionary.

	Args:
	    configfile (str): The full path to the JSON configuration file.

	Returns:
	    dict: A dictionary containing the parsed configuration data.

	Raises:
	    FileNotFoundError: If the specified `configfile` does not exist.
	    json.JSONDecodeError: If the file contains malformed JSON syntax.
	"""
    with open(configfile, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file '{configfile}': {e}")
