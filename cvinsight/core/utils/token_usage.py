from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

import logging

# TokenUsageCallbackHandler to calculate the token usage from LLM Response
class TokenUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.token_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "source": "not_set"
        }

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Extract token usage from the LLM response."""
        # First check for usage_metadata in the generations (LangChain's normalized location)
        token_found = False
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    # Check for usage_metadata (normalized token info across chat providers)
                    if hasattr(gen, "usage_metadata") and gen.usage_metadata:
                        usage = gen.usage_metadata
                        self.token_usage["total_tokens"] = usage.get("total_tokens", 0)
                        self.token_usage["prompt_tokens"] = usage.get("input_tokens", 0)  # normalized name for prompt tokens
                        self.token_usage["completion_tokens"] = usage.get("output_tokens", 0)  # normalized name for completion tokens
                        self.token_usage["source"] = "usage_metadata"
                        token_found = True
                        logging.info(f"Token usage found in usage_metadata: {usage}")
                        return
                    
                    # Check for usage_metadata in generation's message (alternate location)
                    if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata") and gen.message.usage_metadata:
                        usage = gen.message.usage_metadata
                        self.token_usage["total_tokens"] = usage.get("total_tokens", 0)
                        self.token_usage["prompt_tokens"] = usage.get("input_tokens", 0)
                        self.token_usage["completion_tokens"] = usage.get("output_tokens", 0)
                        self.token_usage["source"] = "message_usage_metadata"
                        token_found = True
                        logging.info(f"Token usage found in message usage_metadata: {usage}")
                        return
                        
                    # Fall back to checking in generation_info
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        usage = gen.generation_info.get("token_usage", {})
                        if usage:
                            self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
                            self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0) 
                            self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                            self.token_usage["source"] = "generation_info"
                            token_found = True
        
        # Check for token usage in llm_output (standard location)
        if not token_found and hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.token_usage["source"] = "llm_output"