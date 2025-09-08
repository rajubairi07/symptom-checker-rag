# This is the crucial workaround for the sqlite3 issue on Streamlit Cloud
# It MUST be the very first thing in the file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from vectordb import get_chroma_collection
from rag import rag_pipeline

# --- Secret and Model Configuration ---
# This will get keys from st.secrets when deployed, or from a local .env file.
try:
    # Get secrets from Streamlit's secrets manager
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHAT_MODEL = st.secrets["CHAT_MODEL"]
    EMBED_MODEL = st.secrets["EMBED_MODEL"]
except (KeyError, AttributeError):
    # Fallback for local development if st.secrets doesn't exist or keys are missing
    from config import OPENAI_API_KEY, CHAT_MODEL, EMBED_MODEL


# --- Page Configuration ---
st.set_page_config(
    page_title="💊 Medical RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
with st.sidebar:
    st.header("Database Status")
    try:
        # Pass the API key and model name to the function
        collection = get_chroma_collection(api_key=OPENAI_API_KEY, embed_model=EMBED_MODEL)
        if collection:
            st.success(f"📦 {collection.count()} documents in ChromaDB")
        else:
            st.error("Database initialization failed.")
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        collection = None # Ensure collection is None if there's an error

    st.header("Configuration")
    st.info(f"**Chat Model:** `{CHAT_MODEL}`")
    
    top_k = st.slider(
        "Number of results to retrieve",
        min_value=1, max_value=10, value=5,
        help="Lower this value if you encounter rate limit errors."
    )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared!")


# --- Main App ---
st.title("💊 Medical RAG Assistant")
st.caption("⚠️ Not medical advice. For educational purposes only.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if collection is not None:
                try:
                    # Pass all required arguments to the pipeline
                    answer = rag_pipeline(
                        query=query, 
                        chat_history=st.session_state.messages, 
                        collection=collection,
                        api_key=OPENAI_API_KEY,
                        chat_model=CHAT_MODEL,
                        top_k=top_k
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Cannot process query: The database is not available.")

