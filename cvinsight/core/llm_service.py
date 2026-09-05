"""LLM service for CVInsight."""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Type, Any, Dict, Tuple, List
from pydantic import BaseModel
import httpx
from types import SimpleNamespace

from openai import OpenAI, APITimeoutError
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
            api_key: The API key to use. If None, will use config.TOKENROUTER_API_KEY
        """
        self.model_name = model_name or config.DEFAULT_LLM_MODEL
        self.api_key = api_key or config.TOKENROUTER_API_KEY or os.environ.get("TOKENROUTER_API_KEY")
        self.api_url = config.TOKENROUTER_API_URL
        self.deepseek_model_name = config.DEFAULT_GEN_AI_LLM_MODEL
        self.deepseek_api_key = config.DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key and not config.OPENCODE_ENABLED:
            raise ValueError("TokenRouter API key is required. Either provide it directly to LLMService or set the TOKENROUTER_API_KEY environment variable.")

        if not self.deepseek_api_key:
            raise ValueError("DeepSeek API key is required. Please provide the api key in your environement by set the DEEPSEEK_API_KEY environment variable.")

        if config.OPENCODE_ENABLED and (
            not config.OPENCODE_PROVIDER_ID or not config.OPENCODE_MODEL_ID
        ):
            raise ValueError(
                "OPENCODE_PROVIDER_ID and OPENCODE_MODEL_ID are required "
                "when OpenCode extraction is enabled."
            )

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
        Get a LLM instance for extraction, served by an OpenAI-compatible endpoint.

        Returns:
            A ChatOpenAI instance.
        """
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            model=self.model_name,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_REQUEST_TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    
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
            
            if config.OPENCODE_ENABLED:
                result, token_usage = self._extract_with_opencode(
                    pydantic_model, prompt_template, input_variables, input_data
                )
            else:
                # Use the custom callback to track token usage.
                callback_handler = TokenUsageCallbackHandler()
                chain = self.create_extraction_chain(pydantic_model, prompt_template, input_variables)
                result = chain.invoke(input_data, config={"callbacks": [callback_handler]})
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
            
        except APITimeoutError as e:
            print(f"LLM request timed out after {config.LLM_REQUEST_TIMEOUT}s: {e}")
            # Tag the usage as a timeout so callers can tell it apart from a genuinely empty extraction
            timeout_token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "source": "timeout"}
            return {}, timeout_token_usage
        except Exception as e:
            print(f"Error extracting information with LLM: {e}")
            # Return an empty dictionary and empty token usage
            empty_token_usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "source": "error"}
            return {}, empty_token_usage

    def _extract_with_opencode(
        self,
        pydantic_model: Type[BaseModel],
        prompt_template: str,
        input_variables: list,
        input_data: dict,
    ) -> Tuple[Any, Dict[str, int]]:
        """Create an OpenCode session, send one prompt, and remove the session."""
        parser = JsonOutputParser(pydantic_object=pydantic_model)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=input_variables,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        ).format(**input_data)
        usage = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        session_id = None

        try:
            with httpx.Client(timeout=config.LLM_REQUEST_TIMEOUT) as client:
                session_response = client.post(f"{config.OPENCODE_URL}/session", json={})
                session_response.raise_for_status()
                session_id = session_response.json().get("id")
                if not session_id:
                    raise ValueError("OpenCode did not return a session id")

                response = client.post(
                    f"{config.OPENCODE_URL}/session/{session_id}/message",
                    json={
                        "model": {
                            "providerID": config.OPENCODE_PROVIDER_ID,
                            "modelID": config.OPENCODE_MODEL_ID,
                        },
                        "parts": [{"type": "text", "text": prompt}],
                    },
                )
                response.raise_for_status()
                payload = response.json()

            message = payload.get("info", payload) if isinstance(payload, dict) else {}
            parts = payload.get("parts", []) if isinstance(payload, dict) else []
            text = "".join(
                part.get("text", "") for part in parts
                if isinstance(part, dict) and part.get("type") == "text"
            )
            if not text and isinstance(message, dict):
                text = message.get("text", "")
            if not text:
                raise ValueError("OpenCode returned an empty response")

            response_usage = message.get("tokens", message.get("usage", {})) if isinstance(message, dict) else {}
            if isinstance(response_usage, dict):
                usage["prompt_tokens"] = int(response_usage.get("input", response_usage.get("prompt_tokens", 0)) or 0)
                usage["completion_tokens"] = int(response_usage.get("output", response_usage.get("completion_tokens", 0)) or 0)
                usage["total_tokens"] = int(response_usage.get("total", 0) or 0)
                if not usage["total_tokens"]:
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            result = parser.parse(text)
            if not usage["total_tokens"]:
                usage.update({"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4})
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                usage["is_estimated"] = True
                usage["source"] = "estimation"
            return result, usage
        finally:
            if session_id:
                try:
                    with httpx.Client(timeout=config.LLM_REQUEST_TIMEOUT) as client:
                        client.delete(f"{config.OPENCODE_URL}/session/{session_id}")
                except Exception:
                    logging.warning("Unable to delete OpenCode session %s", session_id, exc_info=True)
        
    def generate_content(self, messages: List[AIMessageResponse], max_token: int = 720) -> Tuple[Any, Dict[str, Any]]:
        token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "is_estimated": True
        }
        
        try:
            if config.OPENCODE_ENABLED:
                return self._generate_content_with_opencode(messages, max_token)

            # @Call tiktoken to estimating the rough token usage
            estimated_token: int = 0
            estimator = TiktokenEstimator()

            for _, val in enumerate(messages):
                estimated_token += estimator.calculate(val.content)

            # Set the default prompt tokens with estimated one
            token_usage["prompt_tokens"] = estimated_token
            token_usage["total_tokens"] = estimated_token

            # Call the LLM model to get the response
            response = self.deepseek_llm.chat.completions.create(
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

    def _generate_content_with_opencode(
        self,
        messages: List[AIMessageResponse],
        max_token: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Generate content through a one-shot OpenCode session."""
        prompt = "\n\n".join(
            f"{message.role}: {message.content}" for message in messages
        )
        token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        session_id = None

        try:
            with httpx.Client(timeout=config.LLM_REQUEST_TIMEOUT) as client:
                session_response = client.post(f"{config.OPENCODE_URL}/session", json={})
                session_response.raise_for_status()
                session_id = session_response.json().get("id")
                if not session_id:
                    raise ValueError("OpenCode did not return a session id")

                response = client.post(
                    f"{config.OPENCODE_URL}/session/{session_id}/message",
                    json={
                        "model": {
                            "providerID": config.OPENCODE_PROVIDER_ID,
                            "modelID": config.OPENCODE_MODEL_ID,
                        },
                        "parts": [{"type": "text", "text": prompt}],
                    },
                )
                response.raise_for_status()
                payload = response.json()

            info = payload.get("info", {}) if isinstance(payload, dict) else {}
            parts = payload.get("parts", []) if isinstance(payload, dict) else []
            content = "".join(
                part.get("text", "") for part in parts
                if isinstance(part, dict) and part.get("type") == "text"
            )
            if not content:
                raise ValueError("OpenCode returned an empty response")

            tokens = info.get("tokens", {}) if isinstance(info, dict) else {}
            token_usage["prompt_tokens"] = int(tokens.get("input", 0) or 0)
            token_usage["completion_tokens"] = int(tokens.get("output", 0) or 0)
            token_usage["total_tokens"] = int(tokens.get("total", 0) or 0)
            if not token_usage["total_tokens"]:
                token_usage["prompt_tokens"] = len(prompt) // 4
                token_usage["completion_tokens"] = len(content) // 4
                token_usage["total_tokens"] = (
                    token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                )
                token_usage["is_estimated"] = True
                token_usage["source"] = "estimation"

            # Keep the existing plugin-facing OpenAI response shape.
            result = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
            return result, token_usage
        except Exception as e:
            print(f"Error generate new content with OpenCode: {e}")
            return {}, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "source": "error"}
        finally:
            if session_id:
                try:
                    with httpx.Client(timeout=config.LLM_REQUEST_TIMEOUT) as client:
                        client.delete(f"{config.OPENCODE_URL}/session/{session_id}")
                except Exception:
                    logging.warning("Unable to delete OpenCode session %s", session_id, exc_info=True)
        
    def generate_content_stream(
        self,
        messages: List[AIMessageResponse],
        max_token: int = 720,
        token_usage_out: Dict[str, Any] = None,
    ):
        """Generate content from DeepSeek with streaming. Yields text chunks."""
        estimated_token = 0
        estimator = TiktokenEstimator()
        for val in messages:
            estimated_token += estimator.calculate(val.content)

        if token_usage_out is not None:
            token_usage_out.update({
                "total_tokens": estimated_token,
                "prompt_tokens": estimated_token,
                "completion_tokens": 0,
                "is_estimated": True,
            })

        try:
            stream = self.deepseek_llm.chat.completions.create(
                model=self.deepseek_model_name,
                messages=[msg.model_dump() for msg in messages],
                stream=True,
                stream_options={"include_usage": True},  # DeepSeek/OpenAI: sends usage in last chunk
                max_tokens=max_token,
                temperature=0.3,
            )

            for chunk in stream:
                # Last chunk has usage populated (from stream_options)
                if token_usage_out is not None and getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                    if usage.total_tokens > 0:
                        token_usage_out.update({
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                            "is_estimated": False,
                        })

                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"Error streaming content with LLM: {e}")
            if token_usage_out is not None:
                token_usage_out.update({"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "source": "error"})
