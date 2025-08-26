import logging
import sys
import os


import logging
import os
import sys

def setup_logging(log_file_path, logger_object_name, level=logging.DEBUG, filemode="a+"):
    """Sets up logging for the application.

    Configures a named logger that writes to both a file and standard output.

    Args:
        log_file_path (str): The path to the log file. The directory part of the
            path will be created if it doesn't exist.
        logger_object_name (str): The name of the logger object. This is a required argument.
        level (int, optional): The logging level. Defaults to logging.INFO.
        filemode (str, optional): The file mode for the log file. Defaults to "a+"
            to append.

    Returns:
        logging.Logger: The configured logger object.

    Raises:
        OSError: If there's an issue creating the log directory.
        TypeError: If logger_object_name is not a string.
        ValueError: If log_file_path is empty.
        FileNotFoundError: If the log file directory cannot be created.
    """

    if not isinstance(logger_object_name, str):
        raise TypeError("logger_object_name must be a string")

    if not log_file_path:
        raise ValueError("log_file_path cannot be empty")

    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            raise FileNotFoundError(f"Could not create log directory: {e}")

    # Create file handler and set level
    file_handler = logging.FileHandler(log_file_path, mode=filemode)
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Create stream handler and set level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    # Get logger and add handlers
    logger = logging.getLogger(logger_object_name)
    logger.setLevel(level)  # Ensure logger itself has the correct level
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def horizontal_line(length, space=2, char="*"):
    """Creates a formatted horizontal line of characters.

	This utility function generates a string consisting of a repeating character
	separated by a specified number of spaces.

	Args:
	    length (int): The number of times the character should appear in the line.
	        If `length` is less than or equal to 0, an empty string is returned.
	    space (int, optional): The number of spaces to place after each character.
	        Defaults to 2.
	    char (str, optional): The single character to use for the line.
	        Defaults to "*".

	Returns:
	    str: A string representing the horizontal line.

	Raises:
	    TypeError: If `length` or `space` are not integers, or if `char` is not a string.
	    ValueError: If `space` is a negative number.
	"""

    if not isinstance(length, int):
        raise TypeError("length must be an integer")
    if not isinstance(space, int):
        raise TypeError("space must be an integer")
    if not isinstance(char, str):
        raise TypeError("char must be a string")
    if space < 0:
        raise ValueError("space cannot be negative")

    if length <= 0:  # Handle cases where length is zero or negative
        return ""

    line = (char + " " * space) * length
    return line.rstrip()