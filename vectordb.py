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

    # Initialize the ChromaDB client - FIRST TRY THE SIMPLE APPROACH
    try:
        # The chroma.sqlite3 file is in the root, so use the path directly
        client = chromadb.PersistentClient(path=local_db_path)
        
        # Get the collection directly - it should exist
        embedding_func = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embed_model
        )
        
        collection = client.get_collection(
            name="disease_symptoms",
            embedding_function=embedding_func
        )
        
        # Verify the collection contents
        count = collection.count()
        st.sidebar.success(f"✅ {count} documents loaded")
        
        return collection
        
    except Exception as e:
        st.error(f"Failed to get collection: {e}")
        
        # Debug: Show what collections actually exist
        try:
            collections = client.list_collections()
            st.sidebar.write(f"Available collections: {[col.name for col in collections]}")
            
            # If disease_symptoms doesn't exist, try the first available collection
            if collections:
                collection = collections[0]
                count = collection.count()
                st.sidebar.warning(f"Using collection '{collection.name}' with {count} documents")
                return collection
                
        except Exception as debug_e:
            st.sidebar.error(f"Debug failed: {debug_e}")
        
        return None