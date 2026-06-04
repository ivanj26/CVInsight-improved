import json
import os
from datetime import datetime, date

from models.requests import work_profile as WorkProfileModel,work_experience as WorkExperienceModel,education as EducationModel, skills as SkillModel
from models.responses import collection_generative as CollectionGenerative, dictionary_generative as DictionaryGenerative

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

        data = {
            "target_role": result.get("current_title"),
            "name": result.get("name"),
            "my_company": "",
            "description": "",
        }
        work_exp = result.get("work_experiences", []) or []
        if len(work_exp) > 0:
            def parse_ddmmyyyy(s: str) -> date:
                return datetime.strptime(s, "%d/%m/%Y").date()

            def key_fn(x: dict[str, any]):
                try:
                    return parse_ddmmyyyy(x.get("start_date", ""))
                except Exception:
                    return date.min  # push bad/missing dates to the beginning

            latest_work_exp = max(work_exp, key=key_fn)
            if latest_work_exp:
                data["my_company"] = latest_work_exp.get("company")
                data["description"] = "\nMy experience: " + ", ".join(latest_work_exp.get("description"))

        # Generate personal summary from given CV/Resume
        summary_result = client.generate_work_profile_recom(
            data,
            log_token_usage=True,
            one_recommendation_only=True
        )

        resp = {"data": result}
        if len(summary_result) > 0:
            resp["personalized_summary"] = summary_result[0]
        
        # Remove the temporary file after extraction finished
        os.remove(temp_path)

        return JSONResponse(content = resp)
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
    
def _sse_stream(generator):
    """Wrap a text-chunk generator into SSE format."""
    try:
        for chunk in generator:
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.post("/generate/work-profile/stream")
def stream_work_profile(payload: WorkProfileModel.WorkProfileRequest):
    return StreamingResponse(
        _sse_stream(client.generate_work_profile_recom_stream(payload.model_dump())),
        media_type="text/event-stream"
    )

@router.post("/generate/work-experience/stream")
def stream_work_experience(payload: WorkExperienceModel.WorkExperienceRequest):
    return StreamingResponse(
        _sse_stream(client.generate_work_exp_recom_stream(payload.model_dump())),
        media_type="text/event-stream"
    )

@router.post("/generate/education/stream")
def stream_education(payload: EducationModel.EducationRequest):
    return StreamingResponse(
        _sse_stream(client.generate_edu_recom_stream(payload.model_dump())),
        media_type="text/event-stream"
    )

@router.post("/generate/skills/stream")
def stream_skills(payload: SkillModel.SkillRequest):
    return StreamingResponse(
        _sse_stream(client.generate_skill_recom_stream(payload.model_dump())),
        media_type="text/event-stream"
    )
