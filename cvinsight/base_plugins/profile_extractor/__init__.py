from typing import Dict, List, Any, Tuple, Type
from pydantic import BaseModel
from ...models.resume_models import ResumeProfile
from ...plugins.base import ExtractorPlugin, PluginMetadata, PluginCategory
import logging

class ProfileExtractorPlugin(ExtractorPlugin):
    """Plugin for extracting profile information from resumes."""
    
    def __init__(self, llm_service=None):
        """
        Initialize the plugin with an LLM service.
        
        Args:
            llm_service: LLM service for extracting information
        """
        self.llm_service = llm_service
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="profile_extractor",
            version="1.0.0",
            description="Extracts basic profile information from resumes",
            category=PluginCategory.BASE,
            author="Resume Analysis Team"
        )
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        logging.info(f"Initializing {self.metadata.name}")
    
    def get_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the extractor."""
        return ResumeProfile
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for the extractor."""
        return """
You are an expert resume parser. Your task is to extract the contact information from the resume text provided below. Specifically, extract the following details:
- Name: Name of the candidate
- Location: Current location of the candidate, if present
- Email: The candidate's email address
- Phone: The candidate's phone number, including country code if present
- LinkedIn: LinkedIn profile URL if present. If present and there is no http/https, prepend the https in front of url
- Current Title: The candidate's current job title if present
- Summary: A brief summary or objective statement if present.

If any of these fields are not present in the resume, return null for that field

Return your output as a JSON object with the below schema
{format_instructions}

Text:
{text}
"""
    
    def get_input_variables(self) -> List[str]:
        """Get the input variables for the prompt template."""
        return ["text"]
    
    def prepare_input_data(self, extracted_text: str) -> Dict[str, Any]:
        """Prepare the input data for the LLM."""
        return {"text": extracted_text}
    
    def extract(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract profile information from text.
        
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
        
        # Process the result
        fields = ["name", "email", "phone", "linkedin", "current_title", "summary"]
        processed_result = {}
        
        if isinstance(result, dict):
            for field in fields:
                processed_result[field] = result.get(field)
        else:
            for field in fields:
                processed_result[field] = getattr(result, field, None)
                
        return processed_result, token_usage