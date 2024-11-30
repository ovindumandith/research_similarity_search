from pymilvus import Collection, connections
from transformers import AutoTokenizer, AutoModel
import torch

# Milvus connection settings
MILVUS_HOST = "localhost"  # Replace with your Milvus server host
MILVUS_PORT = "19530"      # Replace with your Milvus server port
COLLECTION_NAME ="BERT_HNSW"  # Replace with your Milvus collection name

# BERT model and tokenizer initialization
model_name = "sentence-transformers/all-mpnet-base-v2"  # Model used for embedding generation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Connect to Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COLLECTION_NAME)

# Function to generate 2048-dimensional query embeddings
def generate_query_embedding(query, target_dim=2048):
    tokens = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)
    embeddings = torch.mean(output.last_hidden_state, dim=1).squeeze().tolist()
    if len(embeddings) > target_dim:
        return embeddings[:target_dim]
    return embeddings + [0.0] * (target_dim - len(embeddings))

# Function to search for the top_k most similar results
def search_query_in_milvus(query, top_k=5):
    # Generate query embedding
    query_vector = generate_query_embedding(query, target_dim=2048)

    # Set HNSW-specific search parameters
    search_params = {"metric_type": "COSINE", "params": {"ef": 500}}

    # Perform search
    results = collection.search(
        data=[query_vector],  # Input query vector
        anns_field="vector",  # Field name for the vector in the Milvus schema
        param=search_params,  # Search parameters
        limit=top_k           # Number of results to retrieve
    )

    # Process and return results
    matches = []
    for hit in results[0]:
        # Access description directly from the entity
        description = hit.entity['descriptions']  # Access the 'descriptions' field directly
        matches.append({"score": hit.score, "description": description})
    return matches

# Main program loop for user input
if __name__ == "__main__":
    print("=== Milvus Vector Search ===")
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break

        # Retrieve results for the user's query
        try:
            top_matches = search_query_in_milvus(user_query, top_k=5)
            print("\nTop Matches:")
            for i, match in enumerate(top_matches):
                print(f"{i + 1}. Score: {match['score']:.4f}, Description: {match['description']}")
        except Exception as e:
            print(f"An error occurred during search: {e}")
