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
    # The local directory where the database will be stored.
    local_db_path = "chroma_db"

    # Download the database from Hugging Face Hub only if it doesn't exist locally
    if not os.path.exists(local_db_path):
        try:
            # Use a conditional check for st.info to allow this function to run locally
            # from populate_chroma.py without causing an error.
            if "streamlit" in __import__('sys').modules:
                st.info("Downloading ChromaDB database from Hugging Face... this may take a moment.")
            else:
                print("Downloading ChromaDB database from Hugging Face... this may take a moment.")
            
            # Use snapshot_download to reliably download the repository content
            snapshot_download(
                repo_id="rajubairi07/symptom-checker", # Your Hugging Face dataset repo ID
                repo_type="dataset",
                local_dir=local_db_path,
                local_dir_use_symlinks=False # Recommended for cross-platform compatibility
            )

            if "streamlit" in __import__('sys').modules:
                st.success("Database downloaded successfully!")
            else:
                print("Database downloaded successfully!")

        except Exception as e:
            if "streamlit" in __import__('sys').modules:
                st.error(f"Failed to download database from Hugging Face: {e}")
            else:
                print(f"Failed to download database from Hugging Face: {e}")
            return None

    # Initialize the ChromaDB client from the local path
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

