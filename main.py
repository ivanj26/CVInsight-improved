import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from cvinsight import CVInsightClient

# Load API key from .env file if available
load_dotenv()

# Get API key from environment or prompt
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment")

# Initialize the FastAPI app
app = FastAPI()

# Initialize client with API key
client = CVInsightClient(api_key=api_key)

@app.get("/")
def health_check():
    return { "message": "The CVParser service is healthy, running on port 9001!" }

@app.post("/parse")
async def parse_resume(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temp path
        temp_path = f"./temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extract all information (token usage logged separately to logs/ directory)
        result = client.extract_all(temp_path, log_token_usage=True)
        
        # Remove the temporary file after extraction finished
        os.remove(temp_path)

        return JSONResponse(content = {"data": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))