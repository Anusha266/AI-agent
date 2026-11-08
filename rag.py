import json
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
VECTOR_STORE_FILE=os.getenv("VECTOR_STORE_FILE")

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

#ADD embeddings to the chunks
def add_embeddings(chunks):
    for chunk in chunks:
        vector = get_embeddings(chunk["text"])
        chunk["vector"] = vector
        
    return chunks


#takes one file and creates multiple chunks for it. one section=one chunk
def chunk_document(document):
    chunks=[]
    
    chunk_id=0
    for part in document["text"].split("\n\n"):
        part=part.strip()
        if part:
            chunks.append({
                "source":document["source"],
                "chunk_id":chunk_id,
                "text":part
            })
            
            chunk_id+=1
    return chunks
            
    
    
def create_chunks():
    documents=load_documents()
    chunks=[]
    
    for doc in documents:
        chunks.extend(chunk_document(doc))
        
    return chunks
    
def load_vector_store():
    with open(VECTOR_STORE_FILE,"r",encoding='utf-8') as f:
        return json.load(f) #return the chunks
    
    
    
def save_vector_store(chunks):
    with open(VECTOR_STORE_FILE,"w",encoding='utf-8') as f:
        json.dump(chunks,f,indent=4)

def load_chunks():
    if os.path.exists(VECTOR_STORE_FILE):
        chunks=load_vector_store()
        print(f"Loaded {len(chunks)} chunks from the {VECTOR_STORE_FILE}")
    else:
        chunks=create_chunks()
        chunks=add_embeddings(chunks)
        save_vector_store(chunks)
        print(f"we have embedded {len(chunks)} chunks from our Knowledge Base and stored into {VECTOR_STORE_FILE} file")

    return chunks

chunks=load_chunks()


#search method to search topk chunks for a user query
def search(query,top_k=3):
    query_vector=get_embeddings(query)
    
    results=[]
    for chunk in chunks:
        score=cosine_similarity(query_vector,chunk["vector"])
        results.append({
            "score":float(score),
            "source":chunk["source"],
            "chunk_id":chunk["chunk_id"],
            "text":chunk["text"],
            "size":len(chunk["text"])
        })
        
    results.sort(key=lambda rec:rec['score'],reverse=True)
    return results[:top_k]
        
