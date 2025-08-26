import os.path
import sqlite3
from typing import List, Dict, Optional, Any
from Utility.mylogging import setup_logging
from Utility.config_handler import get_nested_config


class DBHandler:
    """Handles database operations for storing and retrieving loss data.

    This class provides methods to create a database, create tables, insert
    training loss data, and retrieve it for analysis. It is designed to work
    with a specific configuration structure, managing a SQLite database file.

    Args:
        config (dict): A dictionary containing the database configuration,
                       including "run_info" with "loss_file_name" (str) and
                       "training" with "losses" (list of str).
    """

    def __init__(self, config: dict, append_path: str = "") -> None:
        """Initializes the DatabaseHandler.

        Args:
            config (dict): A dictionary containing the database configuration.
            append_path (str, optional): An optional path to append to the
                                         database file name. Defaults to "".
        """
        self.db_file_name = os.path.join(append_path, get_nested_config(config, ["run_info", "loss_file_name"]))
        self.losses = get_nested_config(config, ["training", "losses"])
        log_dir = get_nested_config(config, ["run_info", "log_file_name"])

        self.logger = setup_logging(log_file_path=log_dir, logger_object_name="database_handler")

    def create_tables(self):
        """Creates the necessary tables in the database.

        This method calls `create_losses_table` and `create_average_losses_table`
        to set up the database schema.
        """
        self.create_losses_table()
        self.create_average_losses_table()

    def create_losses_table(self, table_name: str):
        """Creates a specified losses table in the database if it doesn't exist.

        The table schema includes columns for epoch, batch, sample_size, a total loss
        column, and a column for each individual loss specified in the configuration.

        Args:
            table_name (str): The name of the table to create.

        Raises:
            sqlite3.Error: If a database error occurs during table creation.
        """
        columns = [
            "epoch INTEGER",
            "batch INTEGER",
            "sample_size INTEGER",
            "loss REAL",
        ]
        columns.extend([f"{loss} REAL" for loss in self.losses])
        query = f'''CREATE TABLE IF NOT EXISTS {table_name} (
                {", ".join(columns)}
            )'''

        try:
            with sqlite3.connect(self.db_file_name) as conn:
                c = conn.cursor()
                c.execute(query)
                conn.commit()
                self.logger.info("Losses table created or already exists.")
        except sqlite3.Error as e:
            error_message = f"Error creating losses table: {e}"
            self.logger.error(error_message)
            raise sqlite3.Error(error_message) from e



    def insert_loss_data(self, table_name: str, epoch: int, batch: int, sample_size: int, loss_values: Dict[str, float]):
        """Inserts a single row of loss data into the specified table.

        The method handles the dynamic creation of the SQL query based on the
        loss names provided during the class initialization.

        Args:
            table_name (str): The name of the table to insert into.
            epoch (int): The epoch number for the data.
            batch (int): The batch number for the data.
            sample_size (int): The number of samples in the current batch.
            loss_values (Dict[str, float]): A dictionary mapping loss names to their
                                            corresponding float values.

        Raises:
            sqlite3.Error: If a database error occurs during data insertion.
        """
        try:
            with sqlite3.connect(self.db_file_name) as conn:
                c = conn.cursor()
                placeholders = ", ".join(["?"] * (len(self.losses) + 4))  # +4 for epoch, batch, sample_size, loss
                insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                data_to_insert = [epoch, batch, sample_size, loss_values["loss"]]  # Start with epoch, batch, loss
                data_to_insert.extend([loss_values[loss] for loss in self.losses])  # Add other losses
                c.execute(insert_query, data_to_insert)
                conn.commit()
        except sqlite3.Error as e:
            error_message = f"Error inserting loss data for epoch={epoch}, batch={batch}: {e}"
            self.logger.error(error_message)
            raise sqlite3.Error(error_message) from e

    def get_loss_data(self,table_name:str, epoch: Optional[int] = None, batch: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """Retrieves loss data from the database.

        Args:
            epoch: Optional epoch to filter by.
            batch: Optional batch to filter by.

        Returns:
            A list of dictionaries, where each dictionary represents a row
            from the losses table. Returns an empty list if no data is found
            or if the database file doesn't exist.  Returns None if there is an error
            accessing the database.
        """
        try:
            with sqlite3.connect(self.db_file_name) as conn:
                c = conn.cursor()
                query = f"SELECT * FROM {table_name}"
                where_clause = []

                if epoch is not None:
                    where_clause.append(f"epoch = {epoch}")
                if batch is not None:
                    where_clause.append(f"batch = {batch}")

                if where_clause:
                    query += " WHERE " + " AND ".join(where_clause)

                c.execute(query)
                rows = c.fetchall()

                if not rows:  # Check if rows is empty
                    self.logger.info("No data found in the losses table.")
                    return []  # Return empty list

                column_names = [description[0] for description in c.description]  # Get column names
                data = []
                for row in rows:
                    row_data = dict(zip(column_names, row))  # Create dictionary for each row
                    data.append(row_data)

                return data

        except sqlite3.Error as e:
            error_message = f"Error retrieving loss data: {e}"
            self.logger.error(error_message)
            raise sqlite3.Error(error_message) from e
        except sqlite3.OperationalError as e:  # Catch file not found
            if "no such table" in str(e):  # Check if the error is "no such table"
                self.logger.warning("Losses table not found. It might not have been created yet.")
                return []  # Return empty list
            elif "unable to open database file" in str(e):
                self.logger.warning(f"Database file not found: {self.db_file_name}")
                return []
            else:
                self.logger.error(f"Operational Error: {e}")
                return None  # Return None to indicate error
        except Exception as e:  # Catch other exceptions
            self.logger.error(f"Error: {e}")
            return None

    def get_loss_data_in_list(self, table_name: str, epoch: Optional[int] = None, batch: Optional[int] = None) -> Optional[
        Dict[str, List[Any]]]:
        """Retrieves loss data from the specified table and returns it as a
        dictionary of lists, where each key corresponds to a column name.

        Args:
            table_name (str): The name of the table to query.
            epoch (Optional[int], optional): An optional epoch number to filter the results.
                                             Defaults to None.
            batch (Optional[int], optional): An optional batch number to filter the results.
                                             Defaults to None.

        Returns:
            Optional[Dict[str, List[Any]]]:
                - A dictionary where keys are column names and values are lists of data.
                - An empty dictionary `{}` if no data is found or the table/file doesn't exist.
                - `None` if a critical error occurs during database access.
        """
        try:
            print(self.db_file_name)
            with sqlite3.connect(self.db_file_name) as conn:
                c = conn.cursor()
                query = f"SELECT * FROM {table_name}"
                where_clause = []

                if epoch is not None:
                    where_clause.append(f"epoch = {epoch}")
                if batch is not None:
                    where_clause.append(f"batch = {batch}")

                if where_clause:
                    query += " WHERE " + " AND ".join(where_clause)

                c.execute(query)
                rows = c.fetchall()

                if not rows:  # Check if rows is empty
                    self.logger.info("No data found in the losses table.")
                    return {}  # Return empty dictionary

                column_names = [description[0] for description in c.description]  # Get column names
                data = {column: [] for column in column_names}  # Initialize dictionary with empty lists

                for row in rows:
                    for column, value in zip(column_names, row):
                        data[column].append(value)

                return data

        except sqlite3.Error as e:
            error_message = f"Error retrieving loss data: {e}"
            self.logger.error(error_message)
            raise sqlite3.Error(error_message) from e
        except sqlite3.OperationalError as e:  # Catch file not found
            if "no such table" in str(e):  # Check if the error is "no such table"
                self.logger.warning("Losses table not found. It might not have been created yet.")
                return {}  # Return empty dictionary
            elif "unable to open database file" in str(e):
                self.logger.warning(f"Database file not found: {self.db_file_name}")
                return {}
            else:
                self.logger.error(f"Operational Error: {e}")
                return None  # Return None to indicate error
        except Exception as e:  # Catch other exceptions
            self.logger.error(f"Error: {e}")
            return None


