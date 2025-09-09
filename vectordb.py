import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import snapshot_download
import time

def get_chroma_collection(api_key: str, embed_model: str):
    """
    Initializes a ChromaDB collection. Downloads the DB from Hugging Face if it doesn't exist.
    Uses huggingface_hub to download the folder content directly.
    """
    local_db_path = "chroma_db"

    if not os.path.exists(local_db_path):
        try:
            if "streamlit" in __import__('sys').modules:
                st.info("Downloading ChromaDB database from Hugging Face...")
            else:
                print("Downloading ChromaDB database from Hugging Face...")
            
            snapshot_download(
                repo_id="rajubairi07/symptom-checker",
                repo_type="dataset",
                local_dir=local_db_path,
                local_dir_use_symlinks=False
            )
            
            # --- ADDED DEBUGGING: List downloaded files ---
            if "streamlit" in __import__('sys').modules:
                st.success("Database download completed. Verifying files...")
                st.write("Files in downloaded directory:")
                for root, _, files in os.walk(local_db_path):
                    for name in files:
                        st.text(os.path.join(root, name))
            # --- END DEBUGGING ---

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

    # --- ADDED DEBUGGING: Check collection count after a delay ---
    if "streamlit" in __import__('sys').modules:
        time.sleep(2) # Give ChromaDB a moment to load the collection fully
        count = collection.count()
        st.info(f"Final check: Collection '{collection.name}' has {count} documents.")
    # --- END DEBUGGING ---

    return collection

