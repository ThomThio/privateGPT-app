from flask import Flask, request, jsonify
import os
import urllib.parse

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
    # Create the folder if it doesn't exist
    os.makedirs(src_folder_path, exist_ok=True)
    # Create a sample.txt file in the source_documents directory
    file_path = os.path.join(src_folder_path, "test.txt")
    with open(file_path, "w") as file:
        file.write("This is a test.")
    # Run the ingest.py command
    os.system('python ingest.py --collection test --project test')
    # Delete the sample.txt file
    os.remove(file_path)
    print("embeddings working")

def model_download():
    url = None
    if model_type == "LlamaCpp":
        url = "https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin"
    elif model_type == "GPT4All":
        url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
    elif model_type == "OpenAI":
        url = "https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"

    folder = "models"
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.join(folder, os.path.basename(parsed_url.path))
    # Check if the file already exists
    if not os.path.exists(filename):
        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        # Run wget command to download the file
        os.system(f"wget {url} -O {filename}")
        global model_path
        model_path = filename
        os.environ['MODEL_PATH'] = filename
        print("model downloaded")

# Starting the app with embedding and llm download
@app.before_first_request
def before_first_request():
    test_embedding()
    model_download()

# Example route
@app.route("/")
def root():
    return {"message": "Hello, the APIs are now ready for your embeds and queries!"}

@app.route("/files/", methods=["POST"])
def create_file():
    try:
        files = request.files.getlist("file")
        for file in files:
            insert_file(file)
        return jsonify({"message": "Files inserted successfully"}), 200
    except Exception as e:
        print("exception", e)
        return "Something went wrong", 500

def insert_file(file):
    # Your file processing logic here
    pass

@app.route("/embed2", methods=["POST"])
def embed2():
    try:
        files = request.files.getlist("files")
        collection_name = request.form.get("collection_name")
        project_name = request.form.get("project_name")
        save_path = f"source_documents/{project_name}"
        saved_files = []

        for file in files:
            try:
                file_path = os.path.join(save_path, file.filename)
                saved_files.append(file_path)
                with open(file_path, "wb") as f:
                    f.write(file.read())
            except:
                print("Error saving file")

            if collection_name is None:
                collection_name = file.filename

        os.system(f'python ingest.py --collection {collection_name} --project source_documents/{project_name}')

        # Delete the contents of the folder
        [os.remove(os.path.join(save_path, file.filename)) or os.path.join(save_path, file.filename) for file in files]

        return jsonify({"message": "Files embedded successfully", "saved_files": saved_files}), 200
    except Exception as e:
        print("exception", e)
        return "Something went wrong", 500

@app.route("/embed", methods=["POST"])
def embed():
    try:
        files = request.files.getlist("files")
        collection_name = request.form.get("collection_name")
        saved_files = []

        for file in files:
            file_path = os.path.join(source_directory, file.filename)
            saved_files.append(file_path)

            with open(file_path, "wb") as f:
                f.write(file.read())

            if collection_name is None:
                collection_name = file.filename

        os.system(f'python ingest.py --collection {collection_name}')

        # Delete the contents of the folder
        [os.remove(os.path.join(source_directory, file.filename)) or os.path.join(source_directory, file.filename) for file
         in files]

        return jsonify({"message": "Files embedded successfully", "saved_files": saved_files}), 200
    except Exception as e:
        print("exception", e)
        return "Something went wrong", 500

@app.route("/retrieve", methods=["POST"])
def query():
    try:
        query_text = request.form.get("query")
        collection_name = request.form.get("collection_name")
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        retriever = db.as_retriever()

        callbacks = [StreamingStdOutCallbackHandler()]

        if model_type == "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        elif model_type == "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        else:
            print(f"Model {model_type} not supported!")
            return "Model not supported", 400

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        res = qa(query_text)
        answer, docs = res['result'], res['source_documents']

        return jsonify({"results": answer, "docs": docs}), 200
    except Exception as e:
        print("exception", e)
        return "Something went wrong", 500

if __name__ == "__main__":
    app.run()
