from vectordb import get_chroma_collection

# This will initialize and create the ./chroma_db folder
collection = get_chroma_collection()
print(f"✅ ChromaDB pre-built with {collection.count()} documents.")
