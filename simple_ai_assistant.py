import anthropic
from dotenv import load_dotenv
import os

from tools import TOOLS,FUNCTION_NAME_TO_FUNCTION_MAP

load_dotenv()#by default looks for .env file 

client=anthropic.Anthropic(
    api_key=os.getenv("API_KEY")
)

system_prompt = "Make Sure to respond in a single line."
messages=[]

while True:
    user_input=input("\nYou Anusha: ")

    if user_input.lower() in ["exit","quit","bye","goodbye"]:
        break
    
    messages.append({"role":"user","content":user_input})

    
    for step in range(int(os.getenv("MAX_STEPS"))):
        response =client.messages.create(
                model=os.getenv("MODEL"),
                messages=messages,
                system=system_prompt,
                tools= TOOLS,
                max_tokens=int(os.getenv("MAX_TOKENS"))
            )
            
        print("Initial response=============", response)
        if response.stop_reason !='tool_use':
            break
        messages.append({
            "role":"assistant",
            "content":response.content
        })
        
        tool_responses=[]
        for content in response.content:
            if content.type=="tool_use":
                func_name=content.name
                func_args=content.input
                try:
                    fn = FUNCTION_NAME_TO_FUNCTION_MAP.get(func_name)
                    result = fn(**func_args) if fn else None
                    result=result.content if hasattr(result,'content') else result
                    print("Tool called: ", func_name, "with args: ", func_args, "Result: ", result)
                except Exception as e:
                    result = f"Error executing tool '{func_name}': {str(e)}"
                
                tool_responses.append({
                    "type":"tool_result",
                        "tool_use_id":content.id,
                            "content":str(result)
                        }
                )
        messages.append({
            "role":"user",
            "content":tool_responses
        })
        
    else:
        print("Maximum steps reached. Ending the conversation.")
        fallback = "Sorry , I couldn't provide a complete response within the maximum steps limit."
        messages.append({"role":"assistant","content":fallback})
        print("Assistant Response 2: ", fallback)
        continue
    messages.append({"role":"assistant","content":response.content})
