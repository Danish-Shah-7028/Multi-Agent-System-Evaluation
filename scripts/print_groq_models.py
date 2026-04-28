import os
import httpx
import json

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

key = os.environ.get('GROQ_API_KEY_1') or os.environ.get('GROQ_API_KEY')
if not key:
    print('GROQ_API_KEY not set')
    raise SystemExit(1)

r = httpx.get('https://api.groq.com/openai/v1/models', headers={'Authorization': f'Bearer {key}'}, timeout=15)
if r.status_code != 200:
    print('Failed to list models:', r.status_code, r.text)
    raise SystemExit(1)

data = r.json()
for m in data.get('data', []):
    print(m.get('id'), '| context_window=', m.get('context_window'))
