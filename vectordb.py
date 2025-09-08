import os
import requests
import zipfile
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# MODIFIED: This function now requires the API key and model name to be passed to it.
@st.cache_resource
def get_chroma_collection(api_key: str, embed_model: str):
    """
    Initializes a ChromaDB collection. Downloads the DB from Hugging Face if it doesn't exist.
    The API key and embedding model name are now required arguments.
    """
    # MODIFIED: Use a reliable, local path within the project directory.
    db_path = "chroma_db"

    # Download and unzip only if the DB is not already present locally
    if not os.path.exists(db_path):
        st.info("Downloading ChromaDB database from Hugging Face... this may take a moment.")
        os.makedirs(db_path, exist_ok=True)
        
        # This should be a direct link to your ZIP file on Hugging Face
        url = "https://huggingface.co/datasets/rajubairi07/symptom-checker/blob/main/chroma_db.zip"
        zip_path = os.path.join(db_path, "chroma_db.zip")

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(db_path)
            
            os.remove(zip_path) # Clean up the downloaded zip file
            st.success("Database downloaded and unzipped successfully!")

        except Exception as e:
            st.error(f"Failed to download or extract database: {e}")
            return None # Return None if download fails

    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(path=db_path)

    # MODIFIED: Use the passed-in arguments to create the embedding function
    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embed_model
    )

    collection = client.get_or_create_collection(
        name="disease_symptoms",
        embedding_function=embedding_func
    )

    return collection
