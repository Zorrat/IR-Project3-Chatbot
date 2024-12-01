
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer
import pickle


# Use a pre-trained SentenceTransformer model to create document embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to save/load embeddings and FAISS index
embeddings_path = "./doc_vectors/doc_embeddings.pkl"
index_path = "./doc_vectors/faiss_index.bin"

# Check if embeddings and index already exist
if os.path.exists(embeddings_path) and os.path.exists(index_path):
    # Load the saved embeddings and FAISS index
    with open(embeddings_path, "rb") as f:
        doc_embeddings = pickle.load(f)
    index = faiss.read_index(index_path)
else:
    # Load the documents from JSON file
    json_path = "data.json"  # Replace with your actual JSON file path
    with open(json_path, "r") as f:
        documents = json.load(f)

    # Flatten the documents into a list for indexing
    document_list = []
    document_titles = []
    for topic, docs in documents.items():
        for doc in docs:
            document_list.append(doc["summary"])
            document_titles.append(doc["title"])
    # Create embeddings and FAISS index if they do not exist
    doc_embeddings = embedding_model.encode(document_list, convert_to_tensor=True).cpu().numpy().astype('float32')

    # Create FAISS index for efficient similarity search
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)

    # Save the embeddings and FAISS index for future use
    with open(embeddings_path, "wb") as f:
        pickle.dump(doc_embeddings, f)
    faiss.write_index(index, index_path)

# Function to retrieve top-k similar documents based on the user query
def retrieve_documents(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().astype('float32')
    faiss.normalize_L2(np.atleast_2d(query_embedding))
    distances, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_docs = [document_list[i] for i in indices[0]]
    retrieved_titles = [document_titles[i] for i in indices[0]]
    return retrieved_titles, retrieved_docs

if __name__ == "__main__":
    print(retrieve_documents("stock market indicators"))
