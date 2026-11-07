import os
from dotenv import load_dotenv
import anthropic
# from tools import TOOLS

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

SYSTEM_PROMPT = '''When answering questions that require tools,
briefly explain what you are going to do before
requesting the tool.
Keep responses under 200 words.'''
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
    
