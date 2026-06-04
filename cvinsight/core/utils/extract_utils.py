import json
import re

def extract_json(text: str):
    match = re.search(r'```json\n([\s\S]*?)\n```', text, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # Assume the entire text is JSON
        json_str = text.strip()

    return json.loads(json_str)