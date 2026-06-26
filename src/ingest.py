import pandas as pd
import sqlite3

import pandas as pd
import sqlite3

def load_raw_to_db(csv_path="data/raw/superstore.csv",
                   db_path="database/superstore.db"):
    """
    Load the raw Superstore CSV into a SQLite database.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.
    db_path : str
        Path to the SQLite database.
    """

    # Load CSV
    df = pd.read_csv(csv_path, encoding="latin1")

    # Connect to SQLite
    conn = sqlite3.connect(db_path)

    # Write to database
    df.to_sql("raw_superstore", conn, if_exists="replace", index=False)

    conn.close()

    print("Raw data successfully loaded into SQLite.")


def run_ingestion(csv_path="data/raw/superstore.csv",
                  db_path="database/superstore.db"):
    """
    Orchestrates the ingestion step.
    Loads raw data and writes it to the SQLite database.
    """
    load_raw_to_db(csv_path, db_path)
    print("Ingestion pipeline completed.")