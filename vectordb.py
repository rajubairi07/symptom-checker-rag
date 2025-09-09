import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download

@st.cache_resource(show_spinner="Loading ChromaDB database...")
def get_chroma_collection(api_key: str, embed_model: str):
    """
    Initializes a ChromaDB collection with caching to avoid repeated downloads.
    """
    local_db_path = "chroma_db"
    
    # Download the database
    try:
        snapshot_download(
            repo_id="rajubairi07/symptom-checker",
            repo_type="dataset",
            local_dir=local_db_path,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        st.error(f"Failed to download database: {e}")
        return None

    try:
        client = chromadb.PersistentClient(path=local_db_path)
        
        # FIRST: Try to get the collection without embedding function
        try:
            collection = client.get_collection(name="disease_symptoms")
            st.sidebar.success("✓ Got collection without embedding function")
        except:
            # SECOND: Try with embedding function if first approach fails
            embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=embed_model
            )
            collection = client.get_collection(
                name="disease_symptoms",
                embedding_function=embedding_func
            )
            st.sidebar.success("✓ Got collection with embedding function")
        
        count = collection.count()
        st.sidebar.success(f"✅ {count} documents loaded")
        return collection
        
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None