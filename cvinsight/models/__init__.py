# Models package

from .resume_models import (
    ResumeProfile,
    ResumeSkills,
    Education,
    ResumeEducation,
    Experience,
    ResumeWorkExperience,
    WorkDates,
    Resume,
    Skills
)

from .content_generation_models import (
    ContentGenerationResponse,
    AIMessageResponse
)

# Alias for backwards compatibility (old code that imports ProfileInfo from models)
ProfileInfo = ResumeProfile

# Import Skills from models.py for backwards compatibility
from . import Skills 