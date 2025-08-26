import sqlite3
import numpy as np
import json


def load_data_from_db(db_name):
    """Loads normalized spectra and radionuclide percentages from a SQLite database.

    Args:
        db_name (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple containing two lists:
            - x_data (list): List of normalized spectra (NumPy arrays).
            - rn_percent (list): List of corresponding radionuclide percentages (NumPy arrays).
             Returns empty lists if database is empty or an error occurs.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute('SELECT id, ps_spectrum, radionuclide_percentage FROM ps_spectra')
        rows = cursor.fetchall()

        x_data = []
        rn_percent = []
        for row in rows:
            _, ps_spectrum_json, radionuclide_percentage_json = row  # Discard id if not needed.
            try:  # Handle JSON decoding errors.
                ps_spectrum = np.array(json.loads(ps_spectrum_json))
                rn_percentage = np.array(json.loads(radionuclide_percentage_json))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in database {db_name}: {e}")
                continue # Skip to next row if JSON is malformed.

            # Min-Max Normalization of the spectrum
            x_min = ps_spectrum.min()
            x_max = ps_spectrum.max()
            if x_max - x_min > 0:
                ps_spectrum_normalized = (ps_spectrum - x_min) / (x_max - x_min)
            else:
                ps_spectrum_normalized = np.zeros_like(ps_spectrum)  # Handle the case where all values are the same.

            x_data.append(ps_spectrum_normalized)
            rn_percent.append(rn_percentage)

        return x_data, rn_percent

    except sqlite3.Error as e:
        print(f"Error reading database {db_name}: {e}")
        return [], []  # Return empty lists in case of an error.
    finally:
        if 'conn' in locals() and conn: # Check if the connection exists before closing.
            conn.close()


def load_data_from_multiple_dbs(db_names):
    """Loads normalized spectra and radionuclide percentages from multiple SQLite databases.

    Args:
        db_names (list): List of database file paths.

    Returns:
        tuple: A tuple containing two lists:
            - x_data (list): Consolidated list of normalized spectra (NumPy arrays) from all databases.
            - rn_percent (list): Consolidated list of corresponding radionuclide percentages (NumPy arrays).
            Returns empty lists if all databases are empty or an error occurs with all databases.
    """
    x_data = []
    rn_percent = []

    for db_name in db_names:
        db_x_data, db_rn_percent = load_data_from_db(db_name)
        x_data.extend(db_x_data)
        rn_percent.extend(db_rn_percent)

    return x_data, rn_percent