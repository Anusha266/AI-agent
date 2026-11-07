
import os
import numpy 


from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()  # by default looks for .env file

client = OpenAI(
    base_url = os.getenv("BASE_URL"),
    api_key = os.getenv("OPENAI_API_KEY")
)

def get_embeddings(text):
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    dot_product = numpy.dot(vec_a, vec_b)
    norm_a = numpy.linalg.norm(vec_a)
    norm_b = numpy.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


text1="python is a programming language"
text2="python is used to write AI related code snippets"
text3="I love eating Apples every morning"

embedding1 = get_embeddings(text1)
embedding2 = get_embeddings(text2)
embedding3 = get_embeddings(text3)

similarity12 = cosine_similarity(embedding1, embedding2)
similarity13 = cosine_similarity(embedding1, embedding3)
similarity23 = cosine_similarity(embedding2, embedding3)

print(f"Similarity between text1 and text2: {similarity12:.2f}")
print(f"Similarity between text1 and text3: {similarity13:.2f}")
print(f"Similarity between text2 and text3: {similarity23:.2f}")