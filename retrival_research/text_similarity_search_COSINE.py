import spacy
from pymilvus import connections, Collection, MilvusException

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Connect to Milvus server
connections.connect(host="localhost", port="19530")

try:
    while True:
        # Get user input for query or exit
        user_query = input("\nEnter your query (or type 'exit' to quit): ")

        # Exit loop if user types 'exit'
        if user_query.lower() == 'exit':
            break

        # Generate embedding from user query
        query_doc = nlp(user_query)
        query_vector = query_doc.vector[:128].tolist()

        # Define search parameters for COSINE similarity
        search_params = {
            "metric_type": "COSINE",  # Set similarity metric to COSINE
            "params": {"nprobe": 10}  # nprobe determines the number of clusters searched
        }

        # Initialize Milvus collection
        collection_name = "MDPCC_cosine"  # Replace with your collection name
        collection = Collection(collection_name)

        # Perform search using COSINE similarity
        print("\nSearch Results using COSINE similarity:")
        similarity_search_result = collection.search(
            data=[query_vector],  # Query vector
            anns_field="vector",  # Search field
            param=search_params,  # Search parameters
            limit=3,  # Number of results to retrieve
            output_fields=['description']  # Retrieve descriptions
        )

        # Display results
        for idx, hit in enumerate(similarity_search_result[0]):
            score = hit.distance
            description = hit.entity.description
            print(f"{idx + 1}. {description} (distance: {score})")

except MilvusException as e:
    print(f"Error occurred: {e}")

finally:
    # Disconnect from Milvus
    connections.disconnect()
