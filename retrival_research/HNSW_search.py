import spacy
from pymilvus import connections, Collection, MilvusException

# Function to pad query vector to match expected dimension
def pad_vector(vector, target_dim):
    if len(vector) < target_dim:
        return vector + [0] * (target_dim - len(vector))
    return vector[:target_dim]  # Truncate if necessary

# Load spaCy model
spacy_model = spacy.load('en_core_web_lg')

# Connect to Milvus server
connections.connect(host="localhost", port="19530")

try:
    while True:
        user_input = input("\nDescribe what cloud computing concept you would like to search\n")
        if user_input.lower() == 'exit':
            break

        # Generate vector
        user_input_doc = spacy_model(user_input)
        user_vector = user_input_doc.vector[:128].tolist()

        # Adjust query vector to match collection dimension
        collection_name = "MDPCC_HNSW"
        collection = Collection(collection_name)
        schema = collection.schema
        vector_field = next(field for field in schema.fields if field.name == "vector")
        expected_dim = vector_field.params["dim"]
        adjusted_user_vector = pad_vector(user_vector, expected_dim)

        # Define search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"efConstruction": 300, "M": 32}
        }

        # Perform similarity search
        similarity_search_result = collection.search(
            data=[adjusted_user_vector],
            anns_field="vector",
            param=search_params,
            limit=3,
            output_fields=['description']
        )

        # Display results
        print("\nSearch Results:")
        for idx, hit in enumerate(similarity_search_result[0]):
            print(f"{idx + 1}. {hit.entity.description} (distance: {hit.distance})")

except MilvusException as e:
    print(f"Milvus Error: {e}")
finally:
    connections.disconnect(alias="default")
