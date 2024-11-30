import tkinter as tk
from tkinter import ttk, messagebox
from transformers import AutoTokenizer, AutoModel
import torch
import scann

# Load pre-trained BERT model and tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to pad/truncate embeddings to a specific dimension
def pad_or_truncate(vector, target_dim):
    if len(vector) > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - len(vector))

# Function to generate embeddings using BERT
def get_bert_embedding(text, target_dim=2048):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens)
    embeddings = torch.mean(output.last_hidden_state, dim=1).squeeze()
    embeddings = embeddings.tolist()
    return pad_or_truncate(embeddings, target_dim)

# Example dataset: Replace with your actual data

# Precompute embeddings for SCANN
target_dim = 2048  # Match your vector dimension
description_vectors = [get_bert_embedding(desc, target_dim=target_dim) for desc in descriptions]

# Build SCANN index
index = scann.ScannBuilder(description_vectors, num_neighbors=5, distance_measure="cosine").build()

# Function to perform SCANN search
def perform_search(query):
    try:
        # Generate query embedding
        query_vector = get_bert_embedding(query, target_dim=target_dim)

        # Perform search using SCANN
        neighbors, distances = index.search(query_vector)

        # Return matched descriptions and distances
        return [(descriptions[i], distances[idx]) for idx, i in enumerate(neighbors)]

    except Exception as e:
        messagebox.showerror("Search Error", f"An error occurred during search: {e}")
        return []

# GUI Implementation
def submit_query():
    query = query_input.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    # Perform the search and display results
    results = perform_search(query)
    result_tree.delete(*result_tree.get_children())  # Clear existing results

    for idx, (description, distance) in enumerate(results):
        result_tree.insert("", "end", values=(idx + 1, description, f"{distance:.4f}"))

# Main Tkinter window
root = tk.Tk()
root.title("Cloud Computing Concept Search (SCANN)")
root.geometry("800x600")

# Query Input
query_frame = ttk.LabelFrame(root, text="Enter Query", padding=(10, 10))
query_frame.pack(fill="x", padx=10, pady=10)

query_input = ttk.Entry(query_frame, width=80)
query_input.pack(side="left", padx=10, pady=10, fill="x", expand=True)

submit_button = ttk.Button(query_frame, text="Search", command=submit_query)
submit_button.pack(side="right", padx=10, pady=10)

# Results Display
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

# Run the Tkinter main loop
root.mainloop()