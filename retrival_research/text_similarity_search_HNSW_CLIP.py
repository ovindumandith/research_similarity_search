import tkinter as tk
from tkinter import ttk, messagebox
from transformers import CLIPProcessor, CLIPModel
import torch
from pymilvus import connections, Collection, FieldSchema, DataType, MilvusException

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate embeddings using CLIP
def get_clip_embedding(text, target_dim=512):  # Default CLIP embedding size is 512
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    embeddings = outputs.squeeze().tolist()
    return pad_or_truncate(embeddings, target_dim)

# Function to pad/truncate embeddings to a specific dimension
def pad_or_truncate(vector, target_dim):
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))

# Milvus connection setup
connections.connect(host="localhost", port="19530", alias="default")

# Function to create HNSW index for CLIP collection
def create_hnsw_index(collection_name):
    try:
        # Create an index for HNSW
        collection = Collection(collection_name)
        index_params = {
            "metric_type": "COSINE",  # Metric for similarity calculation
            "params": {"M": 32, "efConstruction": 500},  # HNSW parameters
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print("HNSW Index created successfully for CLIP collection.")
    except MilvusException as e:
        messagebox.showerror("Index Error", f"An error occurred while creating the index: {e}")

# Function to perform similarity search using CLIP
def perform_search(query):
    try:
        user_vector = get_clip_embedding(query, target_dim=512)

        # Define search parameters for HNSW with COSINE metric
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 500},  # ef for search performance
        }

        # Initialize Milvus collection
        collection_name = "CLIP_HNSW"
        collection = Collection(collection_name)

        # Perform similarity search in Milvus
        search_results = collection.search(
            data=[user_vector],
            anns_field="vector",
            param=search_params,
            limit=5,
            output_fields=["description"]
        )

        return [(hit.entity.description, hit.distance) for hit in search_results[0]]

    except MilvusException as e:
        messagebox.showerror("Search Error", f"An error occurred during search: {e}")
        return []

# Tkinter GUI for CLIP
def submit_query():
    query = query_input.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    results = perform_search(query)
    result_tree.delete(*result_tree.get_children())

    for idx, (description, distance) in enumerate(results):
        result_tree.insert("", "end", values=(idx + 1, description, f"{distance:.4f}"))

root = tk.Tk()
root.title("CLIP Concept Search")
root.geometry("800x600")

query_frame = ttk.LabelFrame(root, text="Enter Query", padding=(10, 10))
query_frame.pack(fill="x", padx=10, pady=10)

query_input = ttk.Entry(query_frame, width=80)
query_input.pack(side="left", padx=10, pady=10, fill="x", expand=True)

submit_button = ttk.Button(query_frame, text="Search", command=submit_query)
submit_button.pack(side="right", padx=10, pady=10)

result_frame = ttk.LabelFrame(root, text="Search Results", padding=(10, 10))
result_frame.pack(fill="both", padx=10, pady=10, expand=True)

result_tree = ttk.Treeview(result_frame, columns=("Rank", "Description", "Distance"), show="headings")
result_tree.heading("Rank", text="Rank")
result_tree.heading("Description", text="Description")
result_tree.heading("Distance", text="Distance")
result_tree.column("Rank", width=50, anchor="center")
result_tree.column("Description", width=550, anchor="w")
result_tree.column("Distance", width=100, anchor="center")
result_tree.pack(fill="both", expand=True)

root.mainloop()
connections.disconnect(alias="default")
