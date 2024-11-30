import tkinter as tk
from tkinter import messagebox
import spacy
from pymilvus import connections, Collection, MilvusException

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Connect to Milvus server
connections.connect(host="localhost", port="19530")

# Function to perform similarity search and display the results
def search_query():
    # Get the query entered by the user
    user_query = query_entry.get()
    
    if not user_query:
        messagebox.showwarning("Input Error", "Please enter a query to search.")
        return

    # Generate embedding for user query
    query_doc = nlp(user_query)
    query_vector = query_doc.vector[:128].tolist()

    # Search parameters for COSINE similarity
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }

    # Initialize Milvus collection
    collection_name = "MDPCC_cosine"  # Replace with your collection name
    collection = Collection(collection_name)

    # Perform search using COSINE similarity
    results_text.delete(1.0, tk.END)  # Clear previous results
    try:
        similarity_search_result = collection.search(
            data=[query_vector],  # Query vector
            anns_field="vector",  # Search field
            param=search_params,  # Search parameters
            limit=3,  # Number of results
            output_fields=["description"]  # Retrieve descriptions
        )

        # Display results
        results_text.insert(tk.END, "Search Results using IP similarity:\n")
        for idx, hit in enumerate(similarity_search_result[0]):
            description = hit.entity.description  # Access description field directly
            score = hit.distance  # Similarity score
            results_text.insert(tk.END, f"{idx + 1}. {description} (distance: {score})\n")
    except MilvusException as e:
        results_text.insert(tk.END, f"Error: {str(e)}\n")

# Create the main window
root = tk.Tk()
root.title("Cloud Computing Concept Similarity Search")
root.geometry("600x400")

# Label for instructions
label = tk.Label(root, text="Enter a cloud computing concept to search for similar descriptions:")
label.pack(pady=10)

# Entry widget for user query
query_entry = tk.Entry(root, width=50)
query_entry.pack(pady=10)

# Button to trigger the search
search_button = tk.Button(root, text="Search", command=search_query)
search_button.pack(pady=10)

# Text widget to display search results
results_text = tk.Text(root, width=70, height=15, wrap=tk.WORD)
results_text.pack(pady=10)

# Start the GUI event loop
root.mainloop()
