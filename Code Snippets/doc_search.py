import os
import numpy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # by default looks for .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

EMBED_MODEL = os.getenv("EMBED_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")

#load the documents 
def load_documents(folder_path=KB_FOLDER):
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(
                    {
                        "source":filename,
                        "text":content,
                    }
                )
    return documents
    
#Get the embedding of a text 
def get_embeddings(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec_a, vec_b):
    dot_product = numpy.dot(vec_a, vec_b)
    norm_a = numpy.linalg.norm(vec_a)
    norm_b = numpy.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

#ADD embeddings to the documents
def add_embeddings(documents):
    for doc in documents:
        vector = get_embeddings(doc["text"])
        doc["vector"] = vector
        
    return documents

def search(query,documents,top_k=3):
    query_vector=get_embeddings(query)
    
    results=[]
    for doc in documents:
        score=cosine_similarity(query_vector,doc["vector"])
        results.append({
            "score":float(score),
            "source":doc["source"],
            "text":doc["text"],
            "size":len(doc["text"])
        })
        
    results.sort(key=lambda rec:rec['score'],reverse=True)
    return results[:top_k]
        
#testing method

def test(user_query,documents,top_k=3):
    print(f"\n user query:{user_query}")
    
    results=search(user_query,documents,top_k)
    
    for r in results:
        print(f" {r['score']:.2f} - {r['source']} - {r['size']} characters")
    
    
documents=load_documents()
documents=add_embeddings(documents)
print(f"we have embedded {len(documents)} documents")

user_query="How many days will I get for maternity leave?"

test(user_query,documents,3)