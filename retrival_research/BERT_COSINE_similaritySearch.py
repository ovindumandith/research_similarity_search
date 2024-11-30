from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, Collection, MilvusException

# Load pre-trained BERT model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # A compact and efficient BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to pad/truncate embeddings to a specific dimension
def pad_or_truncate(vector, target_dim):
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))

# Function to generate embeddings using BERT and ensure the correct dimension
def get_bert_embedding(text, target_dim=384):  # Default target dimension to 384 for this model
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)
    # Use the mean pooling of the last hidden state as the embedding
    embeddings = torch.mean(output.last_hidden_state, dim=1).squeeze()
    print(f"Embedding dimension: {embeddings.shape[0]}")  # Check actual size of the generated embeddings
    embeddings = embeddings.tolist()
    return pad_or_truncate(embeddings, target_dim)

# Connect to Milvus server
connections.connect(host="localhost", port="19530")

try:
    while True:
        # Get user input for query or exit
        user_input = input("\nDescribe the cloud computing concept you would like to search (or type 'exit' to quit): ")

        if user_input.lower() == "exit":
            break

        # Generate embedding using BERT and pad to match Milvus collection schema (e.g., 384 dimensions)
        user_vector = get_bert_embedding(user_input, target_dim=384)

        # Define search parameters for HNSW
        search_params = {
            "HNSW": {"metric_type": "COSINE", "params": {"ef": 500}}  # Higher ef value for exhaustive search
        }

        # Initialize Milvus collection
        collection_name = "MDPCC_HNSW"  # Replace with your collection name
        collection = Collection(collection_name)

        # Perform similarity search in Milvus
        similarity_search_result = collection.search(
            data=[user_vector],
            anns_field="vector",
            param=search_params["HNSW"],  # HNSW search parameters
            limit=10,  # Increase limit to 10 results
            output_fields=["descriptions"]
        )

        # Display search results
        print("\nSearch Results:")
        for idx, hit in enumerate(similarity_search_result[0]):
            score = hit.distance
            descriptions = hit.entity.descriptions
            print(f"{idx + 1}. {descriptions} (distance: {score})")

except MilvusException as e:
    print(f"Error: {e}")

finally:
    # Disconnect from Milvus
    connections.disconnect()
