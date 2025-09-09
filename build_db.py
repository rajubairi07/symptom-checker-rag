import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from utils import load_structured_data 
from config import OPENAI_API_KEY, EMBED_MODEL # Loads from your local .env file

def build_and_populate_db():
    """
    A standalone script to reliably build the ChromaDB from scratch.
    """
    db_path = "chroma_db"
    data_file = os.path.join("data", "Final_Augmented_dataset_Diseases_and_Symptoms.csv")

    # 1. Delete old database to ensure a fresh start
    if os.path.exists(db_path):
        print(f"Deleting existing database at '{db_path}'...")
        shutil.rmtree(db_path)
        print("Old database deleted.")

    # 2. Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    print(f"ChromaDB client initialized at '{db_path}'.")

    # 3. Create embedding function
    embedding_func = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBED_MODEL
    )
    print(f"Using embedding model: {EMBED_MODEL}")

    # 4. Create collection
    collection = client.create_collection(
        name="disease_symptoms",
        embedding_function=embedding_func
    )
    print("Collection 'disease_symptoms' created.")

    # 5. Load data from your CSV
    print(f"Loading data from '{data_file}'...")
    documents, ids = load_structured_data(data_file)
    print(f"Loaded {len(documents)} documents to be added.")

    # 6. Add all documents to the collection
    print("Adding documents to the collection... This may take several minutes.")
    # We add in batches to be safe with memory
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    print("All documents have been added.")

    # 7. Final verification
    count = collection.count()
    print("\n----------------------------------------------------")
    print(f"✅ ChromaDB has been built successfully.")
    print(f"   The collection '{collection.name}' now contains {count} documents.")
    print(f"   You can now upload the '{db_path}' folder to Hugging Face.")
    print("----------------------------------------------------")


if __name__ == "__main__":
    # Make sure you have your OPENAI_API_KEY in your .env file
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found. Please check your .env file.")
    else:
        build_and_populate_db()
