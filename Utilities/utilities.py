import os
from dotenv import load_dotenv

load_dotenv()

START_SEQUENCE="\033[90m" # ANSI escape code for gray color
END_SEQUENCE="\033[0m"
def log(text):
    if os.getenv("DEBUG") == "True":
        print(f"     {START_SEQUENCE}{text}{END_SEQUENCE}")  # Print in gray color

