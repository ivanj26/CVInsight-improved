import os

from models.requests import work_profile as WorkProfileModel,work_experience as WorkExperienceModel,education as EducationModel, skills as SkillModel
from models.responses import collection_generative as CollectionGenerative, dictionary_generative as DictionaryGenerative

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from cvinsight import CVInsightClient

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment")

router = APIRouter(prefix="/ai", tags=["ai"])

# Initialize client with API key
client = CVInsightClient(api_key=api_key)

@router.post("/parse")
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
    
@router.post("/generate/work-profile", response_model=CollectionGenerative.CollectionGenerativeResponse)
def generate_content(payload: WorkProfileModel.WorkProfileRequest):
    try:
        result = client.generate_work_profile_recom(payload.model_dump(), log_token_usage=True)
        response = CollectionGenerative.CollectionGenerativeResponse(
            data=CollectionGenerative.CollectionGenerative(recommendations=result)
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate/work-experience", response_model=CollectionGenerative.CollectionGenerativeResponse)
def generate_content(payload: WorkExperienceModel.WorkExperienceRequest):
    try:
        result = client.generate_work_exp_recom(payload.model_dump(), log_token_usage=True)
        response = CollectionGenerative.CollectionGenerativeResponse(
            data=CollectionGenerative.CollectionGenerative(recommendations=result)
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate/education", response_model=CollectionGenerative.CollectionGenerativeResponse)
def generate_content(payload: EducationModel.EducationRequest):
    try:
        result = client.generate_edu_recom(payload.model_dump(), log_token_usage=True)
        response = CollectionGenerative.CollectionGenerativeResponse(
            data=CollectionGenerative.CollectionGenerative(recommendations=result)
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate/skills", response_model=DictionaryGenerative.DictionaryGenerativeResponse)
def generate_content(payload: SkillModel.SkillRequest):
    try:
        result = client.generate_skill_recom(payload.model_dump(), log_token_usage=True)
        response = DictionaryGenerative.DictionaryGenerativeResponse(
            data=result
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))