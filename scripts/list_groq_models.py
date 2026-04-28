import os
import httpx

key = os.environ.get('GROQ_API_KEY')
if not key:
    print('GROQ_API_KEY not set')
    raise SystemExit(1)

urls = [
    'https://api.groq.com/openai/v1/models',
    'https://api.groq.com/v1/models',
    'https://api.groq.com/models'
]

for url in urls:
    try:
        print('Trying', url)
        r = httpx.get(url, headers={'Authorization': f'Bearer {key}'}, timeout=15)
        print('Status:', r.status_code)
        print(r.text[:400])
    except Exception as e:
        print('Error for', url, str(e))
