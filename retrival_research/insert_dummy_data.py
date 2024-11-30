from pymilvus import Collection
import json

# Connect to Milvus server
collection = Collection(name="MDPCC_HNSW")  # Make sure this is your correct collection name

# Load the data from the generated JSON file
with open('dummy_data_for_milvus.json', 'r') as json_file:
    data = json.load(json_file)

# Separate descriptions and vectors for insertion
descriptions = [entry['description'] for entry in data]
vectors = [entry['vector'] for entry in data]

# Insert data into Milvus (ensure the field names match the collection schema)
collection.insert([vectors, descriptions])

print("Data inserted into Milvus.")
