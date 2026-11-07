
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # by default looks for .env file

client = OpenAI(
    base_url = os.getenv("BASE_URL"),
    api_key = os.getenv("OPENAI_API_KEY")
)

response = client.embeddings.create(
    model=os.getenv("EMBED_MODEL"),
    input=["Hello, world!","python is good language","python is a programming language","python is a snake","python is a programming language that lets you work quickly and integrate systems more effectively"]
)

for i, embedding in enumerate(response.data):
    print(f"Embedding {i}: {embedding.embedding[:10]}... (length: {len(embedding.embedding)})")
