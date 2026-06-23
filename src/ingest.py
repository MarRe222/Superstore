import pandas as pd
import sqlite3


df = pd.read_csv("data/raw/superstore.csv", encoding="latin1")
connect = sqlite3.connect("database/superstore.db")
df.to_sql("orders", connect, if_exists="replace", index=False)
connect.close()
