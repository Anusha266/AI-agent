#standard libraries
import os
from types import SimpleNamespace

#third party libraries
import anthropic
from dotenv import load_dotenv

#local modules
from Utilities.utilities import chat_stream, count_tool_events, log, show_response
from tools import TOOLS, run_tool

load_dotenv()#by default looks for .env file 

client=anthropic.Anthropic(
    api_key=os.getenv("API_KEY")
)

messages=[]

while True:
    user_input=input("\nYou Anusha: ")

    if user_input.lower() in ["exit","quit","bye","goodbye"]:
        break
    
    messages.append({"role":"user","content":user_input})

    try:
        for step in range(int(os.getenv("MAX_STEPS"))):
            

            event_obj = chat_stream(messages)
            
            full_text=''
            if event_obj.stop_reason == 'end_turn':
                for block in event_obj.content:
                    if block.type == "text":
                        full_text+=block.text
                        
                if full_text:
                    messages.append({"role":"assistant","content":full_text})
                break
                            
            
            elif event_obj.stop_reason == 'tool_use':

                tool_use_count = count_tool_events(event_obj)
                log(f"step {step+1}: Model Requested {tool_use_count} tool calls")
                
                messages.append({
                    "role":"assistant",
                    "content":event_obj.content
                })
                tool_responses=[]
                for tool in event_obj.content:
                    if tool.type=="tool_use":
                        tool_obj=SimpleNamespace(
                            name=tool.name,
                            input=tool.input,
                            id=tool.id
                        )
                        run_tool(tool_obj,tool_responses)
                
                if tool_responses:
                    messages.append({
                        "role":"user",
                        "content":tool_responses
                    })
            
            else:
                log(f"Unexpected stop reason: {event_obj.stop_reason}")
                break
        else:
            log("Maximum steps reached. Ending the conversation.")
            fallback = "Sorry , I couldn't provide a complete response within the maximum steps limit."
            messages.append({"role":"assistant","content":fallback})
            log(fallback)
            
    except KeyboardInterrupt:
        print("\nConversation interrupted by user.")
        break  
    
