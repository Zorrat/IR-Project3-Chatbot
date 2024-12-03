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
json_path = "data5.json"  # Replace with your actual JSON file path
with open(json_path, "r") as f:
    documents = json.load(f)

# Flatten the documents into a list for indexing
document_list = []
document_titles = []
document_urls = []

topic_documents = {}

document_metadata = {}  # Store metadata including titles and URLs for topic-based documents

for topic, docs in documents.items():
    topic_documents[topic] = []
    document_metadata[topic] = []
    for doc in docs:
        document_list.append(doc["summary"])
        document_titles.append(doc["title"])
        document_urls.append(doc["url"])
        topic_documents[topic].append(doc["summary"])
        document_metadata[topic].append({"title": doc["title"], "url": doc["url"]})

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
def retrieve_documents(query, topics=None, top_k=3):
    if topics:
        documents_to_search = []
        metadata_to_search = []
        for topic in topics:
            if topic in topic_documents:
                documents_to_search.extend(topic_documents[topic])
                metadata_to_search.extend(document_metadata[topic])
        embeddings = embedding_model.encode(documents_to_search, convert_to_tensor=True).cpu().numpy().astype('float32')
        faiss.normalize_L2(embeddings)
        local_index = faiss.IndexFlatL2(embeddings.shape[1])
        local_index.add(embeddings)
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().astype('float32')
        faiss.normalize_L2(np.atleast_2d(query_embedding))  # Ensure input is 2D for normalization
        distances, indices = local_index.search(np.array([query_embedding]), len(documents_to_search))
        valid_indices = [i for i in indices[0] if i >= 0]
        retrieved_docs = []
        retrieved_titles = []
        retrieved_urls = []
        seen_titles = set()  # To avoid duplicate titles
        for i in valid_indices:
            if metadata_to_search[i]["title"] not in seen_titles:
                retrieved_docs.append(documents_to_search[i])
                retrieved_titles.append(metadata_to_search[i]["title"])
                retrieved_urls.append(metadata_to_search[i]["url"])
                seen_titles.add(metadata_to_search[i]["title"])
            if len(retrieved_docs) == top_k:
                break
    else:
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().astype('float32')
        faiss.normalize_L2(np.atleast_2d(query_embedding))  # Ensure input is 2D for normalization
        distances, indices = index.search(np.array([query_embedding]), len(document_list))  # Retrieve more to avoid duplicates
        # Ensure valid indices and avoid duplicates
        retrieved_docs = []
        retrieved_titles = []
        retrieved_urls = []
        seen_titles = set()  # To avoid duplicate titles
        for i in indices[0]:
            if i >= 0 and document_titles[i] not in seen_titles:
                retrieved_docs.append(document_list[i])
                retrieved_titles.append(document_titles[i])
                retrieved_urls.append(document_urls[i])
                seen_titles.add(document_titles[i])
            if len(retrieved_docs) == top_k:
                break
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

# topic classifier
def topic_classifier(query):
    prompt = f"Classify the given query to the given classes: (Health, Environment, Technology, Economy, Entertainment, Sports, Politics, Education, Travel, Food), if multiple topics apply to this specific query, separating multiple topics with commas. If none of the topic above applys, return \"Other\".  \"Other\" should be used independently without any explaination. Do not choose out of the above defined topics. The query: '{query}'"
    topic = chat_with_model(prompt, max_tokens=512)
    topic_list = [t.strip().split()[0] for t in topic.split(',')]
    return topic_list

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

# Topic selection
st.sidebar.title("Select Topics (Optional)")
topic_options = list(topic_documents.keys())
enable_auto_topic_classification = st.sidebar.checkbox("Enable automatic topic classification")
topic_options.insert(0, "None")  # Add an option for no topic selection
selected_topics = st.sidebar.multiselect("Choose topics for document retrieval", topic_options, default=["None"])
selected_topics = [topic for topic in selected_topics if topic != "None"]
selected_topics = None if len(selected_topics) == 0 else selected_topics

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
        if enable_auto_topic_classification:
            if selected_topics is None:
                # Automatically classify the topic if not selected
                selected_topics = topic_classifier(query)
                # Output classified topics in the chatbot response
                classified_topics_message = f"Automatically classified topics: {', '.join(selected_topics)}"
                with st.chat_message("assistant"):
                    st.markdown(classified_topics_message)
                # Add classified topics to chat history
                #st.session_state.chat_history.append({"role": "assistant", "content": classified_topics_message})
                # Display classified topics in the sidebar
                #st.sidebar.write("Automatically classified topics:", ", ".join(selected_topics))
            # Output classified topics in the chatbot response
            classified_topics_message = f"Automatically classified topics: {', '.join(selected_topics)}"
            with st.chat_message("assistant"):
                st.markdown(classified_topics_message)
            # Add classified topics to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": classified_topics_message})
            # Display classified topics in the sidebar
            st.sidebar.write("Automatically classified topics:", ", ".join(selected_topics))
        retrieved_titles, retrieved_docs, retrieved_urls = retrieve_documents(query, selected_topics)

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
