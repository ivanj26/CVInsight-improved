# This file is kept for backward compatibility
# Most of the functionality has been moved to specialized modules

# Import from new locations for backward compatibility
from .file_utils import read_file, validate_file
from .date_utils import parse_date, calculate_experience
from .logging_utils import setup_logging
from .extract_utils import extract_json