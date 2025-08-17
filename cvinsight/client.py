"""Client interface for CVInsight."""
from typing import Dict, List, Any, Optional, Union
import pathlib
import os

from .core.llm_service import LLMService
from .core.resume_processor import PluginResumeProcessor as ResumeProcessor
from .base_plugins.plugin_manager import PluginManager
from .models.resume_models import (
    ResumeProfile,
    Education,
    Experience,
    Skills
)

class CVInsightClient:
    """Client for CVInsight resume analysis.
    
    This client provides methods for analyzing resumes using Google's Gemini models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the CVInsight client.
        
        Args:
            api_key: Google API key for accessing Gemini models. If None, will look for
                    GOOGLE_API_KEY environment variable
            model_name: The name of the model to use. If None, will use default from config
        """
        # Store API key in environment if provided
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        # Initialize services
        self._llm_service = LLMService(model_name=model_name)
        self._plugin_manager = PluginManager(self._llm_service)
        self._plugin_manager.load_all_plugins()
        self._processor = ResumeProcessor(plugin_manager=self._plugin_manager)
    
    def extract_all(self, file_path: str, log_token_usage: bool = True) -> Dict[str, Any]:
        """
        Extract all information from a resume.
        
        Args:
            file_path: Path to the resume file.
            log_token_usage: Whether to log token usage to a separate file (default: True)
            
        Returns:
            Dictionary containing all extracted information (without token usage data).
        """
        resume = self._processor.process_resume(file_path)
        
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
    
    def extract_profile(self, file_path: str) -> Dict[str, Any]:
        """
        Extract profile information from a resume.
        
        Args:
            file_path: Path to the resume file.
            
        Returns:
            Dictionary containing profile information.
        """
        result = self._processor.process_resume(file_path)
        if hasattr(result, 'name'):
            # Create dictionary from resume fields
            profile_data = {
                'name': result.name,
                'email': result.email,
                'phone': result.contact_number,
                'phone_region': result.contact_number_region,
                'location_country': result.location_country,
                'location_city': result.location_city
            }
            return profile_data
        elif result.get("profile"):
            # Convert to dictionary if it's a Pydantic model
            profile = ResumeProfile(**result.get("profile", {}))
            return profile.model_dump() if hasattr(profile, 'model_dump') else profile.__dict__
        return {}
    
    def extract_education(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract education information from a resume.
        
        Args:
            file_path: Path to the resume file.
            
        Returns:
            List of dictionaries containing education information.
        """
        result = self._processor.process_resume(file_path)
        
        # Handle Resume object
        if hasattr(result, 'educations'):
            return [
                edu.model_dump() if hasattr(edu, 'model_dump') else edu.__dict__
                for edu in result.educations
            ]
        # Handle dict result
        elif result.get("education"):
            return [
                Education(**edu).model_dump() 
                if hasattr(Education(**edu), 'model_dump')
                else Education(**edu).__dict__
                for edu in result.get("education", [])
            ]
        return []
    
    def extract_experience(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract experience information from a resume.
        
        Args:
            file_path: Path to the resume file.
            
        Returns:
            List of dictionaries containing experience information.
        """
        result = self._processor.process_resume(file_path)
        
        # Handle Resume object
        if hasattr(result, 'work_experiences'):
            return [
                exp.model_dump() if hasattr(exp, 'model_dump') else exp.__dict__
                for exp in result.work_experiences
            ]
        # Handle dict result
        elif result.get("experience"):
            return [
                Experience(**exp).model_dump() 
                if hasattr(Experience(**exp), 'model_dump')
                else Experience(**exp).__dict__
                for exp in result.get("experience", [])
            ]
        return []
    
    def extract_skills(self, file_path: str) -> Dict[str, Any]:
        """
        Extract skills information from a resume.
        
        Args:
            file_path: Path to the resume file.
            
        Returns:
            Dictionary containing skills information.
        """
        result = self._processor.process_resume(file_path)
        
        # Handle Resume object
        if hasattr(result, 'skills'):
            return {'skills': result.skills}
        # Handle dict result
        elif result.get("skills"):
            skills = Skills(**result.get("skills", {}))
            return skills.model_dump() if hasattr(skills, 'model_dump') else skills.__dict__
        return {'skills': []}
    
    def extract_years_of_experience(self, file_path: str) -> Optional[str]:
        """
        Extract years of experience from a resume.
        
        Args:
            file_path: Path to the resume file.
            
        Returns:
            String containing years of experience or None if not found.
        """
        result = self._processor.process_resume(file_path)
        
        # Handle Resume object
        if hasattr(result, 'YoE') and result.YoE:
            return result.YoE
        # Handle dictionary result
        elif isinstance(result, dict) and (result.get("YoE") or result.get("years_of_experience")):
            return result.get("YoE") or result.get("years_of_experience")
        
        return None
    
    def generate_work_profile_recom(self, data: Dict[str, Any], log_token_usage: bool = True) -> List[str]:
        """
            Generate AI recommendation for user to fill their professional work profile summary.

        Args:
            data: dictionary of <str, any>
        Returns:
            List of recommendation from AI represented in array of string.
        """
        from .base_plugins.work_profile_recommendator import WorkProfileRecommendator
        
        recommendator = WorkProfileRecommendator(self._llm_service)
        result, token_usage = recommendator.generate(data)

        if log_token_usage and token_usage:
            import json
            import os
            from datetime import datetime

            logs_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)

            # Create a log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(logs_dir, f"{recommendator.metadata.name}_token_usage_{timestamp}.json")
            
            # Save token usage into a separate JSON file
            with open(log_file_path, 'w') as f:
                json.dump({"token_usage": token_usage}, f, indent=2)

        return result['recommendations']
    
    def generate_work_exp_recom(self, data: Dict[str, Any], log_token_usage: bool = True) -> List[str]:
        """
            Generate AI recommendation for user to fill their professional work experience.

        Args:
            data: dictionary of <str, any>
        Returns:
            List of recommendation from AI represented in array of string.
        """
        from .base_plugins.work_exp_recommendator import WorkExperienceRecommendator
        
        recommendator = WorkExperienceRecommendator(self._llm_service)
        result, token_usage = recommendator.generate(data)

        if log_token_usage and token_usage:
            import json
            import os
            from datetime import datetime

            logs_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)

            # Create a log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(logs_dir, f"{recommendator.metadata.name}_token_usage_{timestamp}.json")
            
            # Save token usage into a separate JSON file
            with open(log_file_path, 'w') as f:
                json.dump({"token_usage": token_usage}, f, indent=2)

        return result['recommendations']
    
    def generate_edu_recom(self, data: Dict[str, Any], log_token_usage: bool = True) -> List[str]:
        """
            Generate AI recommendation for user to fill their education description.

        Args:
            data: dictionary of <str, any>
        Returns:
            List of recommendation from AI represented in array of string.
        """
        from .base_plugins.education_recommendator import EducationRecommendator
        
        recommendator = EducationRecommendator(self._llm_service)
        result, token_usage = recommendator.generate(data)

        if log_token_usage and token_usage:
            import json
            import os
            from datetime import datetime

            logs_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)

            # Create a log filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(logs_dir, f"{recommendator.metadata.name}_token_usage_{timestamp}.json")
            
            # Save token usage into a separate JSON file
            with open(log_file_path, 'w') as f:
                json.dump({"token_usage": token_usage}, f, indent=2)

        return result['recommendations']

    def analyze_resume(self, resume_path: Union[str, pathlib.Path], 
                      plugins: Optional[List[str]] = None,
                      log_token_usage: bool = True) -> Dict[str, Any]:
        """
        Analyze a resume with specific plugins.
        
        Args:
            resume_path: Path to the resume file (PDF or DOCX)
            plugins: List of plugin names to use (None for all plugins)
            log_token_usage: Whether to log token usage to a separate file (default: True)
            
        Returns:
            Dictionary with results from selected plugins (without token usage data).
        """
        from .api import analyze_resume as api_analyze_resume
        return api_analyze_resume(resume_path, plugins, log_token_usage)
    
    def list_all_plugins(self) -> List[Dict[str, Any]]:
        """
        List all available plugins.
        
        Returns:
            List of dictionaries containing plugin information.
        """
        return self._plugin_manager.list_plugins()
    
    def list_plugins_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        List plugins by category.
        
        Args:
            category: Category to filter plugins by.
            
        Returns:
            List of dictionaries containing plugin information.
        """
        return self._plugin_manager.list_plugins_by_category(category) 