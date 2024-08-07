import os
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import streamlit as st


# Function to query the vector database
def query_collection(query, n_results):
    client = chromadb.PersistentClient(path="./chroma_db")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )
    collection = client.get_collection(
        name="books_collection",
        embedding_function=openai_ef,
    )
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


# Function to save retrieved chunks to a JSON file
def save_chunks_to_json(results, query):
    json_data = {
        "query": query,
        "chunks": []
    }
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        chunk = {
            "book_title": metadata['book_title'],
            "chunk_index": metadata['chunk_index'],
            "content": doc
        }
        json_data["chunks"].append(chunk)

    with open("retrieved_chunks.json", "w") as f:
        json.dump(json_data, f, indent=4)

    st.success("Retrieved chunks have been saved to retrieved_chunks.json")


# Function to generate a response using OpenAI's API
def generate_response(client, query, context):
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant. Answer the query based on the provided context. in 100 words or less. unless user asks for more detail. Do not refer to any author but just combine the best pieces of the provided content to answer the query. Only answer based on the provided content."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )
    return response


# Main Streamlit app
def main():
    st.title("Book Query System")

    # Initialize OpenAI client
    client = OpenAI()

    # Input for number of chunks to use for context
    n_results = st.number_input("How many chunks would you like to use for context?", min_value=1, value=5)

    # Input for user's query
    query = st.text_input("Enter your query (or 'quit' to exit):")

    if query:
        if query.lower() == 'quit':
            st.stop()

        # Query the collection
        results = query_collection(query, n_results)

        # Save the retrieved chunks to a JSON file
        save_chunks_to_json(results, query)

        # Prepare the context for the AI response
        context = "\n\n".join([
            f"Book: {metadata['book_title']}\nChunk Index: {metadata['chunk_index']}\nContent: {doc}"
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0])
        ])

        # Generate and print the AI response
        st.write("Generating response based on the context...")
        response = generate_response(client, query, context)
        st.write(response)


if __name__ == "__main__":
    main()
