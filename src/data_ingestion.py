"""Data ingestion: raw CSV -> SQLite database -> pandas DataFrame.

The project guidelines require the data to be fetched through SQLite. The raw CSV
is written into a SQLite table once, and every run reads it back with a SQL
``SELECT``. Locations and the table name come from the ``paths`` and
``database`` sections of the config.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: A table name must be a plain SQL identifier, because it is placed directly in the query text.
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

        Raises:
            ValueError: If ``database.table`` is not a plain SQL identifier.
        """
        self.cfg = cfg
        self.csv_path: Path = Path(cfg["paths"]["raw_csv"])
        self.db_path: Path = Path(cfg["paths"]["database"])
        self.table: str = cfg["database"]["table"]
        if not _IDENTIFIER_PATTERN.match(self.table):
            raise ValueError(f"database.table must be a plain identifier (letters, digits, _), got {self.table!r}.")

    def ensure_database(self) -> Path:
        """Create the SQLite database from the CSV if it is missing or a rebuild is requested.

        The database is (re)built when the file does not exist, when the table is
        missing, or when ``database.rebuild`` is true. Otherwise it is left alone.

        Returns:
            Path of the SQLite database.

        Raises:
            FileNotFoundError: If the database must be built and the CSV is missing.
        """
        if self.cfg["database"].get("rebuild", False):
            logger.info("database.rebuild is true: rebuilding %s from the CSV", self.db_path)
        elif self._table_exists():
            logger.info("Using existing SQLite table '%s' in %s", self.table, self.db_path)
            return self.db_path
        else:
            logger.info("SQLite table '%s' not found in %s: building it from the CSV", self.table, self.db_path)

        self._write_csv_to_sqlite()
        return self.db_path

    def load_data(self) -> pd.DataFrame:
        """Read every row of the readings table with a SQL ``SELECT``.

        Logs the number of rows and columns loaded and the null count per column.

        Returns:
            DataFrame with one row per sensor reading.

        Raises:
            FileNotFoundError: If the database does not exist (call ``ensure_database`` first).
            ValueError: If the table is empty or the target column is missing.
        """
        if not self.db_path.is_file():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}. Call ensure_database() first.")

        query = f'SELECT * FROM "{self.table}"'
        with closing(sqlite3.connect(self.db_path)) as connection:
            df = pd.read_sql_query(query, connection)

        if df.empty:
            raise ValueError(f"Table '{self.table}' in {self.db_path} is empty.")
        target = self.cfg["features"]["target"]
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in table '{self.table}'.")

        logger.info("Loaded %d rows x %d columns from SQLite table '%s'", len(df), df.shape[1], self.table)
        nulls = df.isna().sum()
        for column, count in nulls[nulls > 0].items():
            logger.info("Column '%s' has %d nulls (%.1f%%)", column, count, 100 * count / len(df))
        return df

    def _write_csv_to_sqlite(self) -> int:
        """Load the raw CSV and write it to the SQLite table, replacing any existing table.

        Writes in batches of ``database.chunksize`` rows.

        Returns:
            Number of rows written.

        Raises:
            FileNotFoundError: If the raw CSV does not exist.
        """
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Raw CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection:
            df.to_sql(
                self.table,
                connection,
                if_exists="replace",
                index=False,
                chunksize=self.cfg["database"].get("chunksize"),
            )
            connection.commit()

        logger.info("Wrote %d rows from %s to SQLite table '%s'", len(df), self.csv_path.name, self.table)
        return len(df)

    def _table_exists(self) -> bool:
        """Check whether the readings table exists in the SQLite database.

        Returns:
            True if the database file exists and contains the table.
        """
        if not self.db_path.is_file():
            return False
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (self.table,)
            ).fetchone()
        return row is not None
