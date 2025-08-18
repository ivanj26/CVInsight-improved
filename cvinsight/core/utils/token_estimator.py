import tiktoken
from abc import ABC, abstractmethod

# Interface for token estimator
class ITokenEstimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, text: str, model_name: str = "deepseek-chat") -> int:
        pass

# Class estimator using tiktoken
class TiktokenEstimator(ITokenEstimator):
    def __init__(self):
        super().__init__()

    def calculate(self, text: str, model_name = "deepseek-chat") -> int:
        """Estimate token count using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")  # Fallback for DeepSeek
        return len(encoding.encode(text))