import requests

files = [("files", open("general/story1.txt", "rb"))]
data = {"collection_name": "my_collection"}

response = requests.post("http://localhost:8000/embed", files=files, data=data)
print(response.json())