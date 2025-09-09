import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download
import os

@st.cache_resource(show_spinner=False)
def get_chroma_collection(api_key: str, embed_model: str):
    """
    Initializes a ChromaDB collection with caching to avoid repeated downloads.
    """
    local_db_path = "chroma_db"
    
    # Download the database if it doesn't exist
    if not os.path.exists(local_db_path):
        with st.spinner("Downloading ChromaDB database from Hugging Face... this may take a moment."):
            try:
                snapshot_download(
                    repo_id="rajubairi07/symptom-checker",
                    repo_type="dataset",
                    local_dir=local_db_path,
                    local_dir_use_symlinks=False
                )
                st.success("Database downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download database: {e}")
                return None

    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=local_db_path)
        embedding_func = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embed_model
        )
        collection = client.get_or_create_collection(
            name="disease_symptoms",
            embedding_function=embedding_func
        )
        return collection
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None