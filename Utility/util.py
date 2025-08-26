import numpy as np
def check_type(var, expected_type, name="variable", element_type=None):
    """
    Checks if `var` is of type `expected_type` and, if it's a list, checks if all elements match `element_type`.

    Args:
        var (any): The variable to check.
        expected_type (type): The expected type of `var` (e.g., list, int, dict).
        name (str, optional): The variable's name for debugging output.
        element_type (type, optional): If `var` is a list, checks if all elements match `element_type`.

    Raises:
        AssertionError: If the type check fails.
    """
    assert isinstance(var, expected_type), f"Expected {name} to be {expected_type.__name__}, but got {type(var).__name__}"

    if element_type and isinstance(var, list):
        assert all(isinstance(elem, element_type) for elem in var), (
            f"All elements in {name} must be {element_type.__name__}, "
            f"but got {[type(elem).__name__ for elem in var]}"
        )


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a NumPy array so that each row (or the full array if 1D) sums to 1.

    :param arr: NumPy array of shape (N,) or (N, M)
    :return: Normalized NumPy array of the same shape
    """
    arr = np.asarray(arr, dtype=np.float32)  # Ensure it's a NumPy array
    if arr.ndim == 1:
        total = arr.sum()
        if np.isclose(total, 0):
            raise ValueError("Cannot normalize array with sum 0.")
        return arr / total
    elif arr.ndim == 2:
        row_sums = arr.sum(axis=1, keepdims=True)
        if np.any(np.isclose(row_sums, 0)):
            raise ValueError("Cannot normalize rows with sum 0.")
        return arr / row_sums
    else:
        raise ValueError("Only 1D or 2D arrays are supported.")