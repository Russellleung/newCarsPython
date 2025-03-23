import lancedb
from dotenv import dotenv_values

config = dotenv_values(".env")


# --------------------------------------------------------------
# Connect to the database
# --------------------------------------------------------------

uri = "data/lancedb"
db = lancedb.connect(uri)


# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table = db.open_table(config["table"])


# --------------------------------------------------------------
# Search the table
# --------------------------------------------------------------
print(table.to_pandas())
result = table.search(query="motor", query_type="vector").limit(3)
print(result.to_pandas()["text"][0])
