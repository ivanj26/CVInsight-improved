from typing import Dict, Any, Type, List, Tuple
from pydantic import BaseModel
from ...plugins.base import ExtractorPlugin, PluginMetadata, PluginCategory
from ...models import Skills
import logging

class SkillsExtractorPlugin(ExtractorPlugin):
    """Extractor plugin for skills information."""
    
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
            name="skills_extractor",
            version="1.0.0",
            description="Extracts skills from resumes",
            category=PluginCategory.BASE,
            author="Resume Analysis Team"
        )
    
    def initialize(self) -> None:
        """Initialize the plugin."""
        logging.info(f"Initializing {self.metadata.name}")
    
    def get_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the extractor."""
        return Skills
    
    def get_prompt_template(self) -> str:
        """Get the prompt template for the extractor."""
        return """
You are an assistant that extracts a list of skills mentioned in the text below. Only focus on Skills section of the below text and limit to up to 5 skills only.
Return your output as a JSON object with the below schema.
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
        Extract skills information from text.
        
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
                "skills": result.get("skills", [])
            }
        else:
            # If result is a Pydantic model, convert to dict
            processed_result = {
                "skills": getattr(result, "skills", [])
            }
        
        return processed_result, token_usage 