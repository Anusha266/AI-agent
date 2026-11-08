from datetime import datetime
import requests

from Utilities.utilities import log
from rag import search

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"http://wttr.in/{city}?format=3"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text
    
def get_user_city():
    city = "bangalore"  # Default city
    return city


def search_knowledge_base(query):
    results=search(query,top_k=2)
    
    log(f"  Knowledge Base Search:{query}")
    
    for res in results:
        log(f"  {res['score']:.2f} - {res['source']} #{res['chunk_id']}")

    blocks=[]
    for res in results:
        source_file=res["source"]
        chunk_id=res["chunk_id"]
        chunk_text=res["text"]
        header = f"[{source_file} #{chunk_id}]"
        block = f"{header}\n{chunk_text}"
        
        blocks.append(block)
    return "\n\n".join(blocks)
        
        
        
def run_tool(tool_call,tool_responses):
    func_name=tool_call.name
    func_args=tool_call.input
    log(f"  Function name and function args: {func_name} {str(func_args)}")
    try:
        fn = FUNCTION_NAME_TO_FUNCTION_MAP.get(func_name)
        result = fn(**func_args) if fn else None
    except Exception as e:
        result = f"Error executing tool '{func_name}': {str(e)}"
    log(f"   Tool result: {str(result)}")
    tool_responses.append({
            "type":"tool_result",
            "tool_use_id":tool_call.id,
            "content":str(result)
        }
    )
    

TOOLS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time in YYYY-MM-DD HH:MM:SS format",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_city_weather",
        "description": "Get the current weather information for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city to get weather information for, e.g., Hyderabad, London, or New York."
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_user_city",
        "description": "Get the user's city. Call this function when user asks for weather information without specifying a city.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name":"search_knowledge_base",
        "description":"Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, tech, incident "
                            "reports and travel expense policies. Use it for any question about company policies or documents. "
                            "Returns the most relevant passages with their source.",
        "input_schema":{
            "type":"object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
]
FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "get_user_city": get_user_city,
    "search_knowledge_base": search_knowledge_base
}
