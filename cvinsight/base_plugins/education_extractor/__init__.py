from typing import Dict, Any, Type, List, Tuple
from pydantic import BaseModel
from ...plugins.base import ExtractorPlugin, PluginMetadata, PluginCategory
from ...models import ResumeEducation
from datetime import date
import logging

class EducationExtractorPlugin(ExtractorPlugin):
    """Extractor plugin for education information."""
    
    def __init__(self, llm_service):
        """Initialize the plugin with an LLM service."""
        self.llm_service = llm_service
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="education_extractor",
            version="1.0.0",
            description="Extracts education history with institutions, dates, and degrees",
            category=PluginCategory.BASE,
            author="Resume Analysis Team"
        )
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        logging.info(f"Initializing {self.metadata.name}")
    
    def get_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the extractor."""
        return ResumeEducation
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for the extractor."""
        return """
You are an expert resume parser. Your task is to extract education details from the resume text provided below. For each education entry, extract the following details:
- College/School (output as "institution")
- Start Date: If mentioned, convert it into the dd/mm/yyyy format. If the day is missing, default it to "01". If the month is missing, default it to "06". If no start date is mentioned, return null. If you encounter Present then use the current date, i.e. {today}.
- End Date: If mentioned, convert it into the dd/mm/yyyy format. If the day is missing, default it to "01". If the month is missing, default it to "06". If no end date is mentioned, return null. If you encounter Present then use the current date, i.e. {today}.
- Location
- Degree
Only focus on Education section of the below text. Simplify the degree and classify to:
    - Bachelor degree
    - Diploma degree
    - Master degree
    - Doctoral degree
If you cannot find anything, return null.

Return your output as a JSON object with the below schema.
{format_instructions}

Text:
{text}
"""
    
    def get_input_variables(self) -> List[str]:
        """Get the input variables for the prompt template."""
        return ["text", "today"]
    
    def prepare_input_data(self, extracted_text: str) -> Dict[str, Any]:
        """Prepare the input data for the LLM."""
        return {
            "text": extracted_text,
            "today": date.today().strftime("%d/%m/%Y")
        }
    
    def extract(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract education information from text.
        
        Args:
            text: The text to extract information from.
            
        Returns:
            A tuple of (extracted_data, token_usage)
        """
        # Prepare prompt from template
        prompt_template = self.get_prompt_template()
        input_data = self.prepare_input_data(text)
        input_variables = self.get_input_variables()
        model = self.get_model()
        
        # Call LLM service
        result, token_usage = self.llm_service.extract_with_llm(
            model,
            prompt_template,
            input_variables,
            input_data
        )
        
        # Add extractor name to token usage
        token_usage["extractor"] = self.metadata.name
        
        # Process the result to ensure it's a dict with the expected keys
        if isinstance(result, dict):
            processed_result = {
                "educations": result.get("educations", [])
            }
        else:
            # If result is a Pydantic model, convert to dict
            processed_result = {
                "educations": getattr(result, "educations", [])
            }
        
        # Ensure each education entry has expected fields
        for edu in processed_result["educations"]:
            if isinstance(edu, dict):
                # Set default values for missing fields
                edu["institution"] = edu.get("institution") or ""
                edu["start_date"] = edu.get("start_date") or ""
                edu["end_date"] = edu.get("end_date") or ""
                edu["location"] = edu.get("location")  # Can be None
                edu["degree"] = edu.get("degree") or ""
        
        return processed_result, token_usage 