"""High-level API for CVInsight."""
import os
import asyncio
import pathlib
from typing import Union, Dict, List, Any, Optional, Tuple

# Import internal modules
from .core.resume_processor import PluginResumeProcessor as ResumeProcessor
from .core.llm_service import LLMService
from .base_plugins.plugin_manager import PluginManager
from .models.resume_models import (
    ResumeProfile as Profile,
    Education,
    Experience,
    Skills,
    Resume as ResumeData
)

# Global instances for reuse
_llm_service = None
_plugin_manager = None
_processor = None
_api_key = None

def configure(api_key: str, model_name: Optional[str] = None):
    """
    Configure the CVInsight API with credentials.
    
    Args:
        api_key: Google API key for Gemini models
        model_name: Optional model name to use
    """
    global _api_key, _llm_service, _plugin_manager, _processor
    
    # Store api key
    _api_key = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Reset services to use new configuration
    _llm_service = None
    _plugin_manager = None
    _processor = None
    
def _get_llm_service():
    """Get or initialize LLM service."""
    global _llm_service, _api_key
    if _llm_service is None:
        _llm_service = LLMService(api_key=_api_key)
    return _llm_service

def _get_plugin_manager():
    """Get or initialize plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        # Import here to avoid circular imports
        _plugin_manager = PluginManager(_get_llm_service())
        _plugin_manager.load_all_plugins()
    return _plugin_manager

def _get_processor():
    """Get or initialize resume processor."""
    global _processor
    if _processor is None:
        _processor = ResumeProcessor(plugin_manager=_get_plugin_manager())
    return _processor

def extract_all(file_path: str, log_token_usage: bool = True) -> Dict[str, Any]:
    """
    Extract all information from a resume.
    
    Args:
        file_path: Path to the resume file.
        log_token_usage: Whether to log token usage to a separate file (default: True)
        
    Returns:
        Dictionary containing all extracted information (without token usage data).
    """
    resume = _get_processor().process_resume(file_path)
    
    # Create a logs directory if it doesn't exist and we need to log token usage
    if log_token_usage and hasattr(resume, 'token_usage') and resume.token_usage:
        import json
        import os
        from datetime import datetime
        
        logs_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log filename based on the resume filename and timestamp
        file_basename = os.path.basename(file_path)
        file_name = os.path.splitext(file_basename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(logs_dir, f"{file_name}_token_usage_{timestamp}.json")
        
        # Save token usage to a separate JSON file
        with open(log_file_path, 'w') as f:
            json.dump({"token_usage": resume.token_usage}, f, indent=2)
    
    # Return the resume as a dictionary but exclude token_usage and file_path
    if hasattr(resume, 'model_dump'):
        return resume.model_dump(exclude={'token_usage', 'file_path'})
    elif isinstance(resume, dict):
        result = resume.copy()
        result.pop('token_usage', None)
        result.pop('file_path', None)
        return result
    
    return resume

def extract_profile(file_path: str) -> Dict[str, Any]:
    """
    Extract profile information from a resume.
    
    Args:
        file_path: Path to the resume file.
        
    Returns:
        Dictionary containing profile information.
    """
    result = _get_processor().process_resume(file_path)
    if result:
        # Create a profile from the resume fields
        if hasattr(result, 'name'):
            profile_data = {
                'name': result.name,
                'email': result.email,
                'phone': result.contact_number,
                'phone_region': result.contact_number_region,
                'location_city': result.location_city,
                'location_country': result.location_country
            }
            return profile_data
        
        # Create profile from Pydantic model
        profile = Profile(
            name=result.name if hasattr(result, 'name') else "",
            email=result.email if hasattr(result, 'email') else None,
            phone=result.contact_number if hasattr(result, 'contact_number') else None,
            phone_region=result.contact_number_region if hasattr(result, 'contact_number_region') else None,
            location_city=result.location_city if hasattr(result, 'location_city') else None,
            location_country=result.location_country if hasattr(result, 'location_country') else None
        )
        return profile.model_dump() if hasattr(profile, 'model_dump') else profile.__dict__
    return {}

def extract_education(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract education information from a resume.
    
    Args:
        file_path: Path to the resume file.
        
    Returns:
        List of dictionaries containing education information.
    """
    result = _get_processor().process_resume(file_path)
    if result and hasattr(result, 'educations'):
        return [
            edu.model_dump() if hasattr(edu, 'model_dump') else edu.__dict__
            for edu in result.educations
        ]
    return []

def extract_experience(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract experience information from a resume.
    
    Args:
        file_path: Path to the resume file.
        
    Returns:
        List of dictionaries containing experience information.
    """
    result = _get_processor().process_resume(file_path)
    if result and hasattr(result, 'work_experiences'):
        return [
            exp.model_dump() if hasattr(exp, 'model_dump') else exp.__dict__
            for exp in result.work_experiences
        ]
    return []

def extract_skills(file_path: str) -> Dict[str, Any]:
    """
    Extract skills information from a resume.
    
    Args:
        file_path: Path to the resume file.
        
    Returns:
        Dictionary containing skills information.
    """
    result = _get_processor().process_resume(file_path)
    if result and hasattr(result, 'skills'):
        return {'skills': result.skills}
    
    # Create a Skills object and return as dictionary
    skills = Skills(skills=[])
    return skills.model_dump() if hasattr(skills, 'model_dump') else skills.__dict__

def extract_years_of_experience(file_path: str) -> Optional[str]:
    """
    Extract years of experience from a resume.
    
    Args:
        file_path: Path to the resume file.
        
    Returns:
        String containing years of experience or None if not found.
    """
    result = _get_processor().process_resume(file_path)
    if result and hasattr(result, 'YoE'):
        return result.YoE
    return None

async def _analyze_resume_async(resume_path: str, plugins: List[Any]) -> Dict[str, Any]:
    """
    Internal async function to analyze resume with parallel plugin processing.
    
    Args:
        resume_path: Path to the resume file
        plugins: List of plugin instances to use
        
    Returns:
        Dictionary with results from selected plugins
    """
    processor = _get_processor()
    
    # Temporarily replace the plugins
    original_plugins = processor.plugin_manager.plugins.copy()
    processor.plugin_manager.plugins = {p.metadata.name: p for p in plugins}
    
    # Process the resume
    resume = processor.process_resume(resume_path)
    
    # Restore original plugins
    processor.plugin_manager.plugins = original_plugins
    
    if resume:
        # Ensure we return a dictionary, not a Pydantic model
        if hasattr(resume, 'model_dump'):
            return resume.model_dump()
        elif hasattr(resume, '__dict__'):
            return resume.__dict__
        elif hasattr(resume, 'dict'):
            return resume.dict()
            
    return {}

def analyze_resume(resume_path: Union[str, pathlib.Path], 
                   plugins: Optional[List[str]] = None,
                   log_token_usage: bool = True) -> Dict[str, Any]:
    """
    Analyze a resume with specific plugins (runs in parallel).
    
    Args:
        resume_path: Path to the resume file (PDF or DOCX)
        plugins: List of plugin names to use (None for all plugins)
        log_token_usage: Whether to log token usage to a separate file (default: True)
        
    Returns:
        Dictionary with results from selected plugins (without token usage data).
    """
    plugin_manager = _get_plugin_manager()
    
    if plugins:
        # Get specific plugins
        selected_plugins = []
        for plugin_name in plugins:
            plugin = plugin_manager.get_plugin(plugin_name)
            if plugin:
                selected_plugins.append(plugin)
    else:
        # Use all plugins
        selected_plugins = list(plugin_manager.plugins.values())
    
    # If no plugins selected or found, return empty result
    if not selected_plugins:
        return {}
    
    # Run async function in an event loop to ensure parallel processing
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create a new loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Create a new loop if the current one is already running
        new_loop = asyncio.new_event_loop()
        try:
            resume = new_loop.run_until_complete(_analyze_resume_async(str(resume_path), selected_plugins))
        finally:
            new_loop.close()
    else:
        resume = loop.run_until_complete(_analyze_resume_async(str(resume_path), selected_plugins))
    
    # Create a logs directory if it doesn't exist and we need to log token usage
    if log_token_usage and resume.get('token_usage'):
        import json
        import os
        from datetime import datetime
        
        logs_dir = os.path.join(os.getcwd(), 'logs/token_usage')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a log filename based on the resume filename and timestamp
        file_basename = os.path.basename(str(resume_path))
        file_name = os.path.splitext(file_basename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(logs_dir, f"{file_name}_token_usage_{timestamp}.json")
        
        # Save token usage to a separate JSON file
        with open(log_file_path, 'w') as f:
            json.dump({"token_usage": resume.get('token_usage', {})}, f, indent=2)
    
    # Remove token_usage and file_path from the result
    result = resume.copy() if isinstance(resume, dict) else {}
    result.pop('token_usage', None)
    result.pop('file_path', None)
    
    return result

def list_all_plugins() -> List[Dict[str, Any]]:
    """
    List all available plugins.
    
    Returns:
        List of dictionaries containing plugin information.
    """
    return _get_plugin_manager().list_plugins()

def list_plugins_by_category(category: str) -> List[Dict[str, Any]]:
    """
    List plugins by category.
    
    Args:
        category: Category to filter plugins by.
        
    Returns:
        List of dictionaries containing plugin information.
    """
    return _get_plugin_manager().list_plugins_by_category(category)
