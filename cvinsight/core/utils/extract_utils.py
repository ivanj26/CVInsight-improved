import json
import re

def extract_json(text: str):
    match = re.search(r'```json\n([\s\S]*?)\n```', text)
    if not match:
        raise ValueError("No JSON block found")
    return json.loads(match.group(1).strip())