import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv('SPEECH_KEY'))
print(os.getenv('SPEECH_REGION'))
print(os.getenv('O_API_KEY'))
print(os.getenv('endpointDocIntel'))
print(os.getenv('key'))

