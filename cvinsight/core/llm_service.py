"""LLM service for CVInsight."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Type, Any, Dict, Tuple, List
from pydantic import BaseModel

from openai import OpenAI
from ..models.content_generation_models import AIMessageResponse

from . import config
from ..core.utils.token_estimator import TiktokenEstimator
from .utils.token_usage import TokenUsageCallbackHandler

import logging
import os

class LLMService:
    """Service for interacting with LLM API."""
    
    def __init__(self, model_name=None, api_key=None):
        """
        Initialize the LLM service.
        
        Args:
            model_name: The name of the model to use. Defaults to config.DEFAULT_LLM_MODEL.
            api_key: The API key to use. If None, will use config.GOOGLE_API_KEY
        """
        self.model_name = model_name or config.DEFAULT_LLM_MODEL
        self.api_key = api_key or config.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
        self.deepseek_model_name = config.DEFAULT_GEN_AI_LLM_MODEL
        self.deepseek_api_key = config.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required. Either provide it directly to LLMService or set the GOOGLE_API_KEY environment variable.")

        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key is required. Please provide the api key in your environement by set the DEEPSEEK_API_KEY environment variable.")

        self.llm = self._get_llm()
        self.deepseek_llm = self._get_deepseek_llm()

    def _get_deepseek_llm(self):
        """
        Get a LLM instance of DeepSeek to generate new contents/recommendations.

        Returns:
            A OpenAI instance
        """
        return OpenAI(base_url=config.DEEPSEEK_API_URL, api_key=self.deepseek_api_key)

    def _get_llm(self):
        """
        Get a LLM instance.
        
        Returns:
            A ChatGoogleGenerativeAI instance.
        """
        return ChatGoogleGenerativeAI(api_key=self.api_key, model=self.model_name)
    
    def create_extraction_chain(self, pydantic_model: Type[BaseModel], prompt_template: str, input_variables: list):
        """
        Create a chain for extracting information using a language model.
        
        Args:
            pydantic_model: The Pydantic model to use for parsing the output.
            prompt_template: The prompt template to use.
            input_variables: The list of input variables for the prompt template.
            
        Returns:
            A chain that can be used to extract information.
        """
        parser = JsonOutputParser(pydantic_object=pydantic_model)
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables,
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        return prompt | self.llm | parser
    
    def extract_with_llm(self, pydantic_model: Type[BaseModel], prompt_template: str, 
                        input_variables: list, input_data: dict) -> Tuple[Any, Dict[str, int]]:
        """
        Extract information from text using a language model.
        
        Args:
            pydantic_model: The Pydantic model to use for parsing the output.
            prompt_template: The prompt template to use.
            input_variables: The list of input variables for the prompt template.
            input_data: The input data to pass to the prompt template.
            
        Returns:
            A tuple containing:
            - The extracted information as a dictionary
            - A dictionary with token usage information
        """
        try:
            # Initialize token usage
            token_usage = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
            
            # Use the custom callback to track token usage
            callback_handler = TokenUsageCallbackHandler()
            
            # Create the chain and include our callback
            chain = self.create_extraction_chain(pydantic_model, prompt_template, input_variables)
            
            # Invoke the chain with our custom callback
            from langchain.callbacks.manager import CallbackManager
            result = chain.invoke(input_data, config={"callbacks": [callback_handler]})
            
            # Get token usage from callback
            token_usage = callback_handler.token_usage
            
            # Estimate tokens if we couldn't get accurate counts
            if token_usage["total_tokens"] == 0:
                # Estimate based on text length
                prompt_text = prompt_template.format(**input_data)
                # Rough estimate: 4 chars per token
                estimated_prompt_tokens = len(prompt_text) // 4
                estimated_completion_tokens = len(str(result)) // 4
                
                token_usage["prompt_tokens"] = estimated_prompt_tokens
                token_usage["completion_tokens"] = estimated_completion_tokens
                token_usage["total_tokens"] = estimated_prompt_tokens + estimated_completion_tokens
                token_usage["is_estimated"] = True
                token_usage["source"] = "estimation"
                logging.info(f"Token counts are estimated. No token information provided by API.")
            
            # Convert Pydantic model to dictionary (for consistency)
            if isinstance(result, pydantic_model):
                return result.model_dump(), token_usage
            elif isinstance(result, dict):
                return result, token_usage
            elif hasattr(result, "__dict__"):
                return result.__dict__, token_usage
                
            # If we got here, something unexpected happened. Return an empty dict.
            return {}, token_usage
            
        except Exception as e:
            print(f"Error extracting information with LLM: {e}")
            # Return an empty dictionary and empty token usage
            empty_token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "source": "error"}
            return {}, empty_token_usage 
        
    def generate_content(self, messages: List[AIMessageResponse], max_token: int = 720) -> Tuple[Any, Dict[str, Any]]:
        token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "is_estimated": True
        }
        
        try:
            # @Call tiktoken to estimating the rough token usage
            estimated_token: int = 0
            estimator = TiktokenEstimator()

            for _, val in enumerate(messages):
                estimated_token += estimator.calculate(val.content)

            # Set the default prompt tokens with estimated one
            token_usage["prompt_tokens"] = estimated_token
            token_usage["total_tokens"] = estimated_token

            # Call the LLM model to get the response
            llm_model = self.deepseek_llm
            response = llm_model.chat.completions.create(
                model=self.deepseek_model_name,
                messages=[msg.model_dump() for msg in messages],
                stream=False,
                max_tokens=max_token,
                temperature=0.3
            )

            # Check the response from LLM
            if not hasattr(response, "choices"):
                raise KeyError("error: missing generative response from AI LLM")
            
            if len(response.choices) == 0:
                raise ValueError("error: no answer from AI LLM")

            # Get the token usage and override the estimated calculation
            if hasattr(response, "usage") and response.usage is not None:
                total_tokens = response.usage.total_tokens
                if total_tokens > 0:
                    token_usage["prompt_tokens"] = response.usage.prompt_tokens
                    token_usage["completion_tokens"] = response.usage.completion_tokens
                    token_usage["total_tokens"] = total_tokens
                    token_usage["is_estimated"] = False
            
            return response, token_usage
        except Exception as e:
            print(f"Error generate new content with LLM: {e}")
            
            # Return an empty dictionary and empty token usage
            token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "source": "error"}
            return {}, token_usage