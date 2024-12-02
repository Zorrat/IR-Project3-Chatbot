import streamlit as st
from llama_cpp import Llama
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch
from sentence_transformers import SentenceTransformer
import pickle

# Load the quantized Mistral model (q4 variant) from TheBloke
model_path = "./mistral-7b-instruct-v0.2.Q3_K_S.gguf"

llm = Llama.from_pretrained(
    # repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    # filename="Mistral-7B-Instruct-v0.3.IQ1_M.gguf",
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
	filename="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    n_ctx=2048,            # Max sequence length to use
    n_threads=8,           # Number of CPU threads to use
    n_gpu_layers=50,       # Number of layers to offload to GPU
    device = "cuda",
    torch_dtype=torch.float16  # Use mixed precision for faster inference
)

# Load the documents from JSON file
json_path = "data.json"  # Replace with your actual JSON file path
with open(json_path, "r") as f:
    documents = json.load(f)

# Flatten the documents into a list for indexing
document_list = []
document_titles = []
document_urls = []

for topic, docs in documents.items():
    for doc in docs:
        document_list.append(doc["summary"])
        document_titles.append(doc["title"])
        document_urls.append(doc["url"])

# Use a pre-trained SentenceTransformer model to create document embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to save/load embeddings and FAISS index
embeddings_path = "./doc_embeddings.pkl"
index_path = "./faiss_index.bin"

# Check if embeddings and index already exist
if os.path.exists(embeddings_path) and os.path.exists(index_path):
    # Load the saved embeddings and FAISS index
    with open(embeddings_path, "rb") as f:
        doc_embeddings = pickle.load(f)
    index = faiss.read_index(index_path)
else:
 
    # Create embeddings and FAISS index if they do not exist
    doc_embeddings =embedding_model.encode(document_list,convert_to_tensor=True).cpu().numpy().astype('float32')

    # Create FAISS index for efficient similarity search
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)

    # Save the embeddings and FAISS index for future use
    with open(embeddings_path, "wb") as f:
        pickle.dump(doc_embeddings, f)
    faiss.write_index(index, index_path)

# Function to retrieve top-k similar documents based on the user query (CPU only)
def retrieve_documents(query, top_k=3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True,).cpu().numpy().astype('float32')
    faiss.normalize_L2(np.atleast_2d(query_embedding))  # Ensure input is 2D for normalization
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Ensure valid indices
    valid_indices = [i for i in indices[0] if i >= 0]
    retrieved_docs = [document_list[i] for i in valid_indices]
    retrieved_titles = [document_titles[i] for i in valid_indices]
    retrieved_urls = [document_urls[i] for i in valid_indices]
    
    if not retrieved_docs:
        retrieved_titles = ["No relevant documents found."]
        retrieved_docs = ["Sorry, I couldn't find any documents that match your query. Please try another query."]
        retrieved_urls = [""]
    
    return retrieved_titles, retrieved_docs, retrieved_urls



# Summarize a single document using the Llama model
def summarize_document(title, content):
    prompt = f"Summarize the following document titled '{title}':\n{content}"
    summary = chat_with_model(prompt,max_tokens = 512)
    return summary

# Chat with the model
def chat_with_model(user_input, max_tokens=256):
    output = llm(
        f"<s>[INST] {user_input} [/INST]",
        max_tokens=max_tokens,
        stop=["</s>"],
        echo=False
    )
    response = output['choices'][0]['text']
    return response

# Streamlit UI for the chatbot
st.title("Chit-Chat and Document Retrieval Bot using Mistral Model")
st.markdown("This interface allows you to chat with a chatbot and retrieve and summarize documents from a custom vector database.")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question or request document summaries"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if prompt.lower().startswith("query:"):
        # Treat as a query and retrieve relevant documents
        query = prompt[len("query:"):].strip()
        retrieved_titles, retrieved_docs, retrieved_urls = retrieve_documents(query)

        # Display the titles and URLs of the retrieved documents
        titles_message = "Top 3 relevant documents:\n\n" + "\n\n".join([f"[{title}]({url})" for title, url in zip(retrieved_titles, retrieved_urls)])
        with st.chat_message("assistant"):
            st.markdown(titles_message)
        # Add titles to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": titles_message})

        # Summarize each retrieved document separately
        for title, content in zip(retrieved_titles, retrieved_docs):
            if title != "No relevant documents found.":
                summary = summarize_document(title, content)
                summary_message = f"**{title}: **\n{summary}"
                with st.chat_message("assistant"):
                    st.markdown(summary_message)
                # Add each document summary to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": summary_message})
    else:
        # Treat as a chit-chat
        response = chat_with_model(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
