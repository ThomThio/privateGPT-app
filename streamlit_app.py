from dotenv import load_dotenv
import os
import streamlit as st
import requests
from typing import List
import json
import socket
from urllib3.connection import HTTPConnection

API_BASE_URL = os.environ.get("API_BASE_URL")

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

from constants import CHROMA_SETTINGS
import chromadb

def list_of_collections():
    client = chromadb.Client(CHROMA_SETTINGS)
    return (client.list_collections())
    
def main():
    st.title("PrivateGPT App: Document Embedding and Retrieval")
    
    # Document upload section
    st.header("Document Upload")
    files = st.file_uploader("Upload document", accept_multiple_files=True)

    project_name = st.selectbox(label="project",options=["ai_story","general"])
    # collection_name = st.text_input("Collection Name") not working for some reason
    collection_name = "ai_story"

    if st.button("Embed"):
        embed_documents(files, project_name,collection_name)
    
    # Query section
    st.header("Document Retrieval")
    collection_names = get_collection_names()
    selected_collection = st.selectbox("Select a document", collection_names)
    if selected_collection:
        query = st.text_input("Query")
        if st.button("Retrieve"):
            retrieve_documents(query, selected_collection)

def embed_documents(files:List[st.runtime.uploaded_file_manager.UploadedFile], project_name:str,collection_name:str):
    endpoint = f"{API_BASE_URL}/embed2"
    files_data = [("files", file) for file in files]
    data = {"collection_name": collection_name,
            "project_name": project_name
            }

    response = requests.post(endpoint, files=files_data, data=data)
    if response.status_code == 200:
        st.success("Documents embedded successfully!")
    else:
        st.error("Document embedding failed.")
        st.write(response.text)

    print("Completed")


def get_collection_names():

    collections = list_of_collections()
    print(collections)
    return ["ai_story","general"]
    # return [collection.name for collection in collections]



def retrieve_documents(query: str, collection_name: str):
    endpoint = f"{API_BASE_URL}/retrieve"
    data = {"query": query, "collection_name": collection_name}

    print(data)

    # Modify socket options for the HTTPConnection class
    HTTPConnection.default_socket_options = (
        HTTPConnection.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            # (socket.SOL_TCP, socket.TCP_KEEPIDLE, 45),
            (socket.SOL_TCP, socket.TCP_KEEPINTVL, 10),
            (socket.SOL_TCP, socket.TCP_KEEPCNT, 6)
        ]
    )
    
    response = requests.post(endpoint, params=data)
    if response.status_code == 200:
        result = response.json()
        st.subheader("Results")
        st.text(result["results"])
        
        st.subheader("Documents")
        for doc in result["docs"]:
            st.text(doc)
    else:
        st.error("Failed to retrieve documents.")
        st.write(response.text)


if __name__ == "__main__":
    main()