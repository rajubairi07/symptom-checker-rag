import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download

@st.cache_resource(show_spinner="Downloading ChromaDB database from Hugging Face...")
def get_chroma_collection(api_key: str, embed_model: str):
    """
    Initializes a ChromaDB collection with caching to avoid repeated downloads.
    """
    local_db_path = "chroma_db"
    
    # Download the database (cached by Streamlit)
    try:
        # This will only download once and cache the result
        snapshot_download(
            repo_id="rajubairi07/symptom-checker",
            repo_type="dataset",
            local_dir=local_db_path,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        st.error(f"Failed to download database from Hugging Face: {e}")
        return None

    # Initialize the ChromaDB client from the local path
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
        
        # Verify the collection has documents
        count = collection.count()
        st.sidebar.success(f"✅ {count} documents loaded in ChromaDB")
        
        return collection
        
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None