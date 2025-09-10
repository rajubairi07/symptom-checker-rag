from openai import OpenAI


def rag_pipeline(
    query: str, 
    chat_history: list, 
    collection, # The initialized ChromaDB collection
    api_key: str, 
    chat_model: str, 
    top_k: int = 5
):


    client = OpenAI(api_key=api_key)

    retrieved = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = retrieved["documents"][0]
    context = "\n".join(docs)
    
    system_prompt = """You are a helpful AI medical assistant.
Base your answers on the retrieved context, but also consider past conversation.
⚠️ Disclaimer: This is not medical advice. For educational purposes only.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})

    response = client.chat.completions.create(
        model=chat_model,
        messages=messages
    )
    
    return response.choices[0].message.content.strip()
