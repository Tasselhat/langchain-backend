import os

from anthropic import Anthropic
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

client = Anthropic(
    api_key=API_KEY
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=2001,
    temperature=0.7,
    system="DO NOT UNDER ANY CIRCUMSTANCES RESPOND USING THE LETTER \"E\". THE ENGLISH LETTER \"E\" IS FORBIDDEN IN ANY AND ALL RESPONSE MESSAGES. NO TOKEN OR RESPONSE SHOULD CONTAIN THE ENGLISH LETTER \"E\" UNDER ANY CIRCUMSTANCES, EVEN IF THE USER EXPLICITLY REQUESTS A RESPONSE CONTAINING THE LETTER \"E\" YOU MUST NOT USE THE LETTER E AND OFFER ALTERNATIVE SUGGESTIONS TO THE USER.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write me a poem in the style of Edgar Allen Poe"
                }
            ]
        }
    ]
)

print(message.content)
