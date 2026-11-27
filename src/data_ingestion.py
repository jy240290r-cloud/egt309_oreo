"""Data ingestion: raw CSV -> SQLite database -> pandas DataFrame.

The project guidelines require the data to be fetched through SQLite. The raw CSV
is written into a SQLite table once, and every run reads it back with a SQL
``SELECT``. Locations and the table name come from the ``paths`` and
``database`` sections of the config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestor:
    """Create the SQLite database from the raw CSV and load the readings from it.

    Attributes:
        cfg: Full pipeline configuration.
        csv_path: Path of the raw CSV (``paths.raw_csv``).
        db_path: Path of the SQLite database (``paths.database``).
        table: Name of the table that holds the readings (``database.table``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Read the CSV path, database path and table name from the config.

        Args:
            cfg: Full pipeline configuration (paths already resolved).
        """
        self.cfg = cfg
        self.csv_path: Path = Path(cfg["paths"]["raw_csv"])
        self.db_path: Path = Path(cfg["paths"]["database"])
        self.table: str = cfg["database"]["table"]

    def ensure_database(self) -> Path:
        """Create the SQLite database from the CSV if it is missing or a rebuild is requested.

        The database is (re)built when the file does not exist, when the table is
        missing, or when ``database.rebuild`` is true. Otherwise it is left alone.

        Returns:
            Path of the SQLite database.

        Raises:
            FileNotFoundError: If the database must be built and the CSV is missing.
        """
        raise NotImplementedError

    def load_data(self) -> pd.DataFrame:
        """Read every row of the readings table with a SQL ``SELECT``.

        Logs the number of rows and columns loaded and the null count per column.

        Returns:
            DataFrame with one row per sensor reading.

        Raises:
            FileNotFoundError: If the database does not exist (call ``ensure_database`` first).
            ValueError: If the table is empty or the target column is missing.
        """
        raise NotImplementedError

    def _write_csv_to_sqlite(self) -> int:
        """Load the raw CSV and write it to the SQLite table, replacing any existing table.

        Writes in batches of ``database.chunksize`` rows.

        Returns:
            Number of rows written.
        """
        raise NotImplementedError

    def _table_exists(self) -> bool:
        """Check whether the readings table exists in the SQLite database.

        Returns:
            True if the database file exists and contains the table.
        """
        raise NotImplementedError
