import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

START_SEQUENCE="\033[90m" 
END_SEQUENCE="\033[0m"
def log(text):
    if os.getenv("DEBUG") == "True":
        print(f"     {START_SEQUENCE}{text}{END_SEQUENCE}")  # Print in gray color

def show_response(text):
    print(text,end='',flush=True)

def count_tool_events(event_obj):
    tool_use_count = 0
    for block in event_obj.content:
        if block.type == "tool_use":
            tool_use_count += 1
        
    return tool_use_count

SYSTEM_PROMPT = '''"You are a helpful, humble assistant for Zorvex Technologies employees. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use get_current_time, get_city_weather and get_user_city only for the current time, the weather, or the user's city. "
            "For ANY question about Zorvex policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses, "
            "you MUST first call search_knowledge_base. "
            "Answer company questions ONLY from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. "
            "After the answer, cite the source like (leave_policy.txt #3). "
            "If the returned text does not contain the answer, say exactly: I could not find this in the knowledge base. "
            "Keep answers under 100 words."'''
client=anthropic.Anthropic(
        api_key=os.getenv("API_KEY")
    )

def chat_stream(messages):
    from tools import TOOLS

    with client.messages.stream(
        model=os.getenv("MODEL"),
        messages=messages,
        system=SYSTEM_PROMPT,
        tools= TOOLS,
        max_tokens=int(os.getenv("MAX_TOKENS"))
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    show_response(event.delta.text)
        final_message = stream.get_final_message()
    
    return final_message
    
