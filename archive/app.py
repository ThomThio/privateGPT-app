import os
import urllib.parse
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from typing import List, Optional
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
source_directory = os.environ.get('SOURCE_DIRECTORY', '../source_documents')
ai_story_directory = os.environ.get('SOURCE_DIRECTORY', '../source_documents/ai_story')

from constants import CHROMA_SETTINGS


def test_embedding():
    src_folder_path = "../source_documents"
    os.makedirs(src_folder_path, exist_ok=True)
    file_path = os.path.join(src_folder_path, "test.txt")
    with open(file_path, "w") as file:
        file.write("This is a test.")
    os.system('python ingest.py --collection test --project test')
    os.remove(file_path)
    print("embeddings working")


def model_download():
    url = None
    match model_type:
        case "LlamaCpp":
            url = "https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
        case "GPT4All":
            url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
        case "OpenAI":
            url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"

    folder = "models"
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.join(folder, os.path.basename(parsed_url.path))
    if os.path.exists(filename):
        print("File already exists.")
        return
    os.makedirs(folder, exist_ok=True)
    os.system(f"wget {url} -O {filename}")
    global model_path
    model_path = filename
    os.environ['MODEL_PATH'] = filename
    print("model downloaded")


@app.route("/")
def root():
    return jsonify({"message": "Hello, the APIs are now ready for your embeds and queries!"})


@app.route("/files/", methods=["POST"])
def create_file():
    file = request.files.get("file")
    fileb = request.files.get("fileb")
    token = request.form.get("token")

    return jsonify({
        "file_size": len(file.read()),
        "token": token,
        "fileb_content_type": fileb.content_type,
    })


@app.route("/embed2", methods=["POST"])
def embed2():
    files = request.files.getlist("files")
    collection_name = request.form.get("collection_name")
    project_name = request.form.get("project_name")

    print("Running embed2")
    save_path = f"source_documents/{project_name}"
    saved_files = []

    for file in files:
        try:
            file_path = os.path.join(save_path, secure_filename(file.filename))
            print(file_path)
            saved_files.append(file_path)
            file.save(file_path)
        except Exception as e:
            print(f"Error saving file: {str(e)}")

    if collection_name is None:
        collection_name = files[0].filename

    os.system(f'python ingest.py --collection {collection_name} --project source_documents/{project_name}')

    for file in files:
        os.remove(os.path.join(save_path, secure_filename(file.filename)))

    return jsonify({"message": "Files embedded successfully", "saved_files": saved_files})


@app.route("/embed", methods=["POST"])
def embed():
    files = request.files.getlist("files")
    collection_name = request.form.get("collection_name")

    print("Running embed1")
    saved_files = []

    for file in files:
        file_path = os.path.join(source_directory, secure_filename(file.filename))
        saved_files.append(file_path)
        file.save(file_path)

    if collection_name is None:
        collection_name = files[0].filename

    os.system(f'python ingest.py --collection {collection_name}')

    for file in files:
        os.remove(os.path.join(source_directory, secure_filename(file.filename)))

    return jsonify({"message": "Files embedded successfully", "saved_files": saved_files})


@app.route("/retrieve", methods=["POST"])
def query():
    query = request.form.get("query")
    collection_name = request.form.get("collection_name")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    callbacks = [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            return jsonify({"error": f"Model {model_type} not supported!"})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    res = qa(query)
    answer, docs = res['result'], res['source_documents']

    return jsonify({"results": answer, "docs": docs})

if __name__ == '__main__':
    API_BASE_URL = os.environ.get("API_BASE_URL")
    app.run(host='localhost',port=8000,debug=True)