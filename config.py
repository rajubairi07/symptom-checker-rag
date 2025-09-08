import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")  
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small") 
