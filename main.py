from src.ingest import run_ingestion
from src.clean import run_cleaning
from src.features import run_features

if __name__ == "__main__":
    run_ingestion("data/raw/superstore.csv", "database/superstore.db")
    run_cleaning("database/superstore.db")
    run_features("database/superstore.db")

    print("Pipeline complete. Run 'python dashboard/app.py' to start the dashboard.")