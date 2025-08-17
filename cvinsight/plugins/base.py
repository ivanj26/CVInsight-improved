from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Type, List, Tuple, Optional
from pydantic import BaseModel

from ..models.content_generation_models import AIMessageResponse

class PluginCategory(Enum):
    """Categories of plugins."""
    BASE = auto()      # Core functionality plugins
    CUSTOM = auto()    # User-created plugins

class PluginMetadata:
    """Metadata for plugins."""
    def __init__(self, name: str, version: str, description: str, 
                 category: PluginCategory = PluginCategory.CUSTOM,
                 author: Optional[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.category = category
        self.author = author
        
class BasePlugin(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""
        pass

class RecommendatorPlugin(BasePlugin):
    """Base class for recommendator plugins."""

    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for the extractor."""
        pass

    @abstractmethod
    def prepare_input_data(self, data: Dict[str, Any]) -> List[AIMessageResponse]: 
        """ Prepare the input data for the LLM """
        pass

    @abstractmethod
    def generate(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate new content from Generative LLM AI.
        
        Args:
            text: The text to extract information from.
            
        Returns:
            A tuple of (output_data, token_usage)
        """
        pass    

class ExtractorPlugin(BasePlugin):
    """Base class for extractor plugins."""
    
    @abstractmethod
    def get_model(self) -> Type[BaseModel]:
        """Get the Pydantic model for the extractor."""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for the extractor."""
        pass
    
    @abstractmethod
    def get_input_variables(self) -> List[str]:
        """Get the input variables for the prompt template."""
        pass
    
    @abstractmethod
    def prepare_input_data(self, extracted_text: str) -> Dict[str, Any]:
        """Prepare the input data for the LLM."""
        pass
    
    @abstractmethod
    def extract(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract information from text.
        
        Args:
            text: The text to extract information from.
            
        Returns:
            A tuple of (extracted_data, token_usage)
        """
        pass 