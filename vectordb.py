import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download

def get_chroma_collection(api_key: str, embed_model: str):
    """
    Initializes a ChromaDB collection. Downloads the DB from Hugging Face if it doesn't exist.
    Uses huggingface_hub to download the folder content directly.
    """
    local_db_path = "chroma_db"

    if not os.path.exists(local_db_path):
        try:
            # This check ensures st.info only runs when in a Streamlit app,
            # preventing errors if this function were ever called from a regular script.
            if "streamlit" in __import__('sys').modules:
                st.info("Downloading ChromaDB database from Hugging Face...")
            
            # Use snapshot_download to reliably download the repository content
            snapshot_download(
                repo_id="rajubairi07/symptom-checker", # Your Hugging Face dataset repo ID
                repo_type="dataset",
                local_dir=local_db_path,
                local_dir_use_symlinks=False
            )
            
            if "streamlit" in __import__('sys').modules:
                st.success("Database download completed.")

        except Exception as e:
            if "streamlit" in __import__('sys').modules:
                st.error(f"Failed to download database from Hugging Face: {e}")
            return None

    # Initialize the ChromaDB client from the local path
    client = chromadb.PersistentClient(path=local_db_path)

    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embed_model
    )

    # get_or_create_collection is safer than create_collection as it won't error
    # if the collection already exists from a previous run.
    collection = client.get_or_create_collection(
        name="disease_symptoms",
        embedding_function=embedding_func
    )

    return collection

