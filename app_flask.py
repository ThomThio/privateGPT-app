import flask
from flask import Flask
from flask import request
from flask_cors import CORS
from llama_index import StorageContext, load_index_from_storage, SimpleDirectoryReader, GPTVectorStoreIndex
import os

from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app)


def insert_file(self, file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(self.KB, filename)
    file.save(filepath)
    document = SimpleDirectoryReader(input_files=[filepath]).load_data()[0]
    self.index.insert(document)

@app.route("/")
def home():
    return "Hello World!"


@app.route("/query")
def query():
    search = request.args.get("search")
    result = ""#gptService.query(search)
    return flask.jsonify({"completion": result}), 200


@app.route("/file/upload", methods=["POST"])
def file_upload():
    try:
        if "file" not in request.files:
            return "Please upload file", 400
        file = request.files["file"]
        # gptService.insert_file(file)
        return "file inserted", 200
    except Exception as e:
        print("exception", e)
        return "Something went wrong", 500


@app.route("/knowledge-base")
def get_kb():
    return "KB"


if __name__ == "__main__":
    app.run(host="localhost", port=5601)