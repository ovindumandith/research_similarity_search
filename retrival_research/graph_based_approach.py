import spacy
import networkx as nx
from pymilvus import connections, Collection, MilvusException
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Connect to Milvus server
connections.connect(host="localhost", port="19530")

# Initialize Graph
graph = nx.Graph()

# Function to perform similarity search and graph building
def search_and_build_graph(user_query):
    # Generate embedding for the user query
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
    similarity_search_result = collection.search(
        data=[query_vector],  # Query vector
        anns_field="vector",  # Search field
        param=search_params,  # Search parameters
        limit=5,  # Number of results to retrieve
        output_fields=['description', 'id']  # Retrieve descriptions and ids
    )

    # Add results to graph and build edges based on similarity score
    for idx, hit in enumerate(similarity_search_result[0]):
        score = hit.distance
        description = hit.entity.description
        doc_id = hit.entity.id  # Retrieve document ID
        print(f"{idx + 1}. {description} (distance: {score})")

        # Add node to graph (description as the node, ID as a unique identifier)
        graph.add_node(doc_id, description=description, score=score)

        # Add edges between the query node and the retrieved document nodes
        graph.add_edge(user_query, doc_id, weight=score)

    # Visualize the graph (optional)
    visualize_graph()

# Function to visualize the graph using networkx and matplotlib
def visualize_graph():
    # Draw the graph with labels
    pos = nx.spring_layout(graph)  # Choose layout for graph
    labels = nx.get_node_attributes(graph, 'description')
    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=6, font_weight='bold')
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight='bold')
    plt.title("Document Similarity Graph")
    plt.show()

# GUI-like interactive loop for querying
try:
    while True:
        # Get user input for query or exit
        user_query = input("\nEnter your query (or type 'exit' to quit): ")

        # Exit loop if user types 'exit'
        if user_query.lower() == 'exit':
            break

        # Search and build graph based on the query
        search_and_build_graph(user_query)

except MilvusException as e:
    print(f"Error occurred: {e}")

finally:
    # Disconnect from Milvus
    connections.disconnect()
