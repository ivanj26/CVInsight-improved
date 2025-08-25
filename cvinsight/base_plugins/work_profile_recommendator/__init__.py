from ...core import LLMService
from ...core.utils import extract_json
from string import Template
from typing import Dict, Any, List, Tuple

from ...plugins.base import RecommendatorPlugin, PluginMetadata, PluginCategory
from ...models.content_generation_models import AIMessageResponse

import logging

class WorkProfileRecommendator(RecommendatorPlugin):
    """ Recommendator plugin for work profile summary information. """

    def __init__(self, llm_service: LLMService = None):
        """
        Initialize the plugin with the LLM service.

        Args:
            llm_service: LLM service for recommend and generate new content.
        """
        self.llm_service = llm_service

    def initialize(self) -> None:
        """Initialize the plugin."""
        logging.info(f"Initializing {self.metadata.name}")

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="work_profile_recommendator",
            version="1.0.0",
            description="Generate recommendation about work profile summary from context information",
            category=PluginCategory.BASE,
        )

    def get_prompt_template(self) -> str:
        """ Get the prompt template for the recommendator """
        return """
My target role: $target_role.
My name: $name
My current work is $role at $my_company.
Description (if exists): $description

Give me at least 8 examples to showcase my own professional profile or you can rephrase description if exists, please make it concise by at max 2 paragraphs for each recommendation
and returns in JSON format with this structure: {\"recommendations\": [\"item1\", \"item2\"]}.
"""

    def prepare_input_data(self, data: Dict[str, Any]) -> List[AIMessageResponse]: 
        """ Prepare the input data for the LLM """
        return [
            AIMessageResponse(role="system", content="You are a CV/resume assistant. **Always returns in JSON format**"),
            AIMessageResponse(
                role="user",
                content=Template(self.get_prompt_template()).safe_substitute(data),
            )
        ]
    
    def generate(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate new content from Generative LLM AI.
        
        Args:
            text: The text to extract information from.
            
        Returns:
            A tuple of (extracted_data, token_usage)
        """
        input_data = self.prepare_input_data(data)

        # Call the LLM service
        result, token_usage = self.llm_service.generate_content(input_data)
        
        # Add extractor name to token usage
        token_usage["extractor"] = self.metadata.name

        if not result:
            return {}, token_usage

        return extract_json(result.choices[0].message.content), token_usage