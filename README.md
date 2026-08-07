# CVInsight

AI-powered resume parsing and analysis using Google's Gemini models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CVInsight is a Python package that helps streamline the resume review process by automatically extracting key information from PDF and DOCX resumes. The system uses Google's Gemini models to process and extract structured data from unstructured resume text through a flexible plugin architecture.

## Key Features

- **Clean Dictionary Output**: All functions return clean Python dictionaries, not Pydantic objects.
- **Separate Token Usage Logging**: Token usage data is logged to separate files in the `logs/` directory.
- **Consistent API**: Both client and API approaches produce the same output format.
- **Easy Authentication**: Multiple ways to provide API keys (direct, environment variables, .env).
- **Dictionary Output**: All extractors return dictionaries or lists of dictionaries for easy JSON serialization.
- **Extract structured information** from resumes (PDF, DOCX).
- **Parse personal details**, education, experience, skills, and more.
- **Customize extraction** with plugins.
- **CLI tool** for batch processing.
- **Plugin-Based Architecture**: Easily extend functionality by adding new plugins.
- **Multiple Resume Formats**: Supports both PDF and DOCX resume file formats.
- **Profile Extraction**: Extracts basic information like name, contact number, and email.
- **Skills Analysis**: Identifies skills from resumes.
- **Education History**: Extracts educational qualifications with institution names, dates, and degrees.
- **Work Experience**: Analyzes professional experience with company names, roles, and dates.
- **Years of Experience**: Calculates total professional experience based on work history.
- **Concurrent Processing**: Processes multiple aspects of resumes in parallel for efficiency.
- **Structured Output**: Provides results in clean, structured JSON format.
- **Token Usage Tracking**: Monitors and logs API token consumption for each resume processed.
- **Separated Log Files**: Keeps resume outputs clean by storing token usage data in separate log files.
- **Automatic Log Rotation**: Implements log rotation to keep log files manageable.
- **Configurable Log Retention**: Automatically cleans up token usage logs after a configurable period.

## Installation

```bash
pip install cvinsight
```

## Quick Start

### Using the Client Interface (Recommended)

```python
from cvinsight import CVInsightClient

# Initialize with your API key
client = CVInsightClient(api_key="YOUR_GEMINI_API_KEY")

# Extract all information from a resume (token usage logged to separate file)
result = client.extract_all("path/to/resume.pdf")
print(result)  # Clean dictionary output without token usage data

# Or extract specific components (all return dictionaries)
profile = client.extract_profile("path/to/resume.pdf")
education = client.extract_education("path/to/resume.pdf")
experience = client.extract_experience("path/to/resume.pdf")
skills = client.extract_skills("path/to/resume.pdf")
yoe = client.extract_years_of_experience("path/to/resume.pdf")
```

### Using the API (Alternative)

```python
import cvinsight

# Configure the API with your credentials
cvinsight.api.configure(api_key="YOUR_GEMINI_API_KEY")

# Extract information from a resume (token usage logged to separate file)
result = cvinsight.extract_all("path/to/resume.pdf")
profile = cvinsight.extract_profile("path/to/resume.pdf")
education = cvinsight.extract_education("path/to/resume.pdf")
```

### Complete Example

```python
import os
import json
from dotenv import load_dotenv
from cvinsight import CVInsightClient

# Load API key from .env file if available
load_dotenv()

# Get API key from environment or prompt
api_key = os.environ.get("TOKENROUTER_API_KEY")
if not api_key:
    api_key = input("Enter your TokenRouter API key: ")

# Initialize client with API key
client = CVInsightClient(api_key=api_key)
resume_path = "path/to/resume.pdf"

# Extract and print years of experience
print("Years of experience:", client.extract_years_of_experience(resume_path))

# Extract and print skills as formatted JSON
skills = client.extract_skills(resume_path)
print("\nSkills:")
print(json.dumps(skills, indent=2))

# Extract all information (token usage logged separately to logs/ directory)
result = client.extract_all(resume_path, log_token_usage=True)
print("\nFull resume information:")
print(json.dumps(result, indent=2))
```

## Configuration

### API Key

You can set the API key in multiple ways:

1.  **Directly in Code (as shown in Quick Start) (Not recommended)**
2.  **Environment Variable:**
    ```bash
    # In your shell
    export TOKENROUTER_API_KEY="YOUR_TOKENROUTER_API_KEY"
    ```
3.  **`.env` File:** Create a `.env` file in your project directory (Recommended):
    ```
    TOKENROUTER_API_KEY=YOUR_TOKENROUTER_API_KEY
    ```

### Other Configuration Options

You can configure the following options in the `.env` file (primarily for development and advanced usage):

- `TOKENROUTER_API_URL`: Base URL of the OpenAI-compatible endpoint (default: https://api.tokenrouter.com/v1)
- `DEFAULT_LLM_MODEL`: Model name to use (default: deepseek-v4-flash)
- `RESUME_DIR`: Directory containing resume files (default: ./Resumes)
- `OUTPUT_DIR`: Directory for processed results (default: ./Results)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `LOG_FILE`: Path to log file
- `TOKEN_LOG_RETENTION_DAYS`: Number of days to keep token usage logs (default: 7)
- `LOG_MAX_SIZE_MB`: Maximum size of log files before rotation in MB (default: 5)
- `LOG_BACKUP_COUNT`: Number of backup log files to keep (default: 3)
- `DEBUG`: Enable or disable debug mode (default: False)


## Command Line Usage

```bash
# Process a resume with all plugins
cvinsight --resume path/to/resume.pdf

# List available plugins
cvinsight --list-plugins

# Process with specific plugins
cvinsight --resume path/to/resume.pdf --plugins profile_extractor,skills_extractor

# Output as JSON
cvinsight --resume path/to/resume.pdf --json

# Save to specific directory
cvinsight --resume path/to/resume.pdf --output ./results
```

For development and advanced usage, `main.py` supports additional arguments:

```bash
# Process a single resume file (using main.py directly)
python main.py --resume example.pdf

# Only display token usage report for a previously processed resume
python main.py --resume example.pdf --report-only

# Specify a custom directory for token usage logs
python main.py --log-dir ./custom_logs

# Enable verbose logging
python main.py --verbose

# Clean up __pycache__ directories and compiled Python files
python main.py --cleanup
```

Example CLI output:
```
Resume Analysis Results:
Name: JOHN DOE
Email: john.doe@example.com
Skills: Python, SQL, Data Analysis, Machine Learning...

Education:
- Bachelor of Science in Computer Science at Example University

Experience:
- Software Engineer at Tech Company
Years of Experience: 5 Years
```

## Token Usage Logging

By default, token usage information is logged to separate files to keep the output data clean.

```python
# Enable token usage logging (default)
result = client.extract_all("resume.pdf", log_token_usage=True)

# Disable token usage logging if needed
result = client.extract_all("resume.pdf", log_token_usage=False)
```

Token usage logs are saved to the `logs/` directory with filenames that include the resume name and timestamp:
```
logs/token_usage/resume_name_token_usage_YYYYMMDD_HHMMSS.json
```

The system tracks token usage for each resume processed and provides:
- A summary report in the console output (when using `main.py`)
- Detailed JSON log files in the `logs/token_usage` directory
- Breakdown of token usage by plugin/extractor

### Token Usage Log Example

```json
{
  "resume_file": "John_Doe.pdf",
  "processed_at": "20250323_031534",
  "token_usage": {
    "total_tokens": 7695,
    "prompt_tokens": 7410,
    "completion_tokens": 285,
    "by_extractor": {
      "profile": {
        "total_tokens": 1445,
        "prompt_tokens": 1423,
        "completion_tokens": 22,
        "source": "message_usage_metadata" // Or "plugins" depending on context
      },
      "skills": {
        "total_tokens": 1383,
        "prompt_tokens": 1304,
        "completion_tokens": 79,
        "source": "message_usage_metadata"
      },
      "education": {
        "total_tokens": 1672,
        "prompt_tokens": 1624,
        "completion_tokens": 48,
        "source": "message_usage_metadata"
      },
      "experience": {
        "total_tokens": 1704,
        "prompt_tokens": 1586,
        "completion_tokens": 118,
        "source": "message_usage_metadata"
      },
      "yoe": {
        "total_tokens": 1491, // Example, actual could be 0 if calculated
        "prompt_tokens": 1473,
        "completion_tokens": 18,
        "source": "message_usage_metadata" // Or "calculated"
      }
    },
    "source": "plugins" // Or another top-level source
  }
}
```

## Dictionary Output

All methods in both the client and API interfaces return clean dictionaries or lists of dictionaries, making it easy to work with the extracted data and convert it to JSON:

```python
# Extract skills as dictionary
skills = client.extract_skills("resume.pdf")
print(skills)
# Output: {'skills': ['Python', 'Machine Learning', 'Data Analysis', ...]}

# Extract education as list of dictionaries
education = client.extract_education("resume.pdf")
print(education)
# Output: [{'degree': 'Bachelor of Science...', 'institution': 'University...', ...}]
```

## Example Output

### Complete Resume JSON Output

```json
{
  "name": "John Doe",
  "contact_number": "+1-123-456-7890",
  "email": "john.doe@example.com",
  "skills": [
    "Python",
    "Machine Learning",
    "Data Analysis",
    "SQL",
    "JavaScript"
  ],
  "educations": [
    {
      "institution": "University of Example",
      "start_date": "2015-09",
      "end_date": "2019-05",
      "location": "Boston, MA",
      "degree": "Bachelor of Science in Computer Science"
    }
  ],
  "work_experiences": [
    {
      "company": "Tech Company Inc.",
      "start_date": "2019-06",
      "end_date": "2023-03",
      "location": "San Francisco, CA",
      "role": "Software Engineer"
    }
  ],
  "YoE": "4 years",
  "file_name": "john_doe.pdf"
}
```

### Skills Output

```json
{
  "skills": [
    "Python",
    "Machine Learning",
    "Data Analysis",
    "SQL",
    "JavaScript"
  ]
}
```

### Education Output

```json
[
  {
    "institution": "University of Example",
    "start_date": "2015-09",
    "end_date": "2019-05",
    "location": "Boston, MA",
    "degree": "Bachelor of Science in Computer Science"
  }
]
```

## Plugin Architecture

The application uses a modular, plugin-based architecture:

- **Plugin Manager**: Discovers, loads, and manages plugins
- **Base Plugin**: Abstract base class for all plugins
- **Built-in Plugins**: Profile, Skills, Education, Experience, and YoE extractors
- **Custom Plugins**: Add your own plugins in the `custom_plugins` directory
- **Plugin Resume Processor**: Processes resumes using the loaded plugins
- **LLM Service**: Centralized service for interacting with language models

For detailed documentation about the plugin architecture and creating custom plugins, please refer to our [Plugin System Wiki Page](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Plugin-System).

### Creating Custom Plugins

You can create custom plugins by inheriting from the `BasePlugin` class and implementing the required methods:

1. Create a new Python file in the `custom_plugins` directory
2. Import the `BasePlugin` class from `base_plugins.base`
3. Create a class that inherits from `BasePlugin`
4. Implement the required abstract methods: `name`, `version`, `description`, `category`, `get_model`, `get_prompt_template`, and `process_output`
5. Add your plugin to the `__all__` list in `custom_plugins/__init__.py`

Check out the [Examples and Tutorials](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Examples-and-Tutorials) wiki page for more examples on how to create and use custom plugins.


## Setup (for Development)

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your API keys
6. Place resume files (PDF or DOCX format) in the `Resumes/` directory (or configure `RESUME_DIR` in `.env`)

## Documentation

- **Wiki Pages**: For detailed documentation, examples, and guides, please visit our [Wiki](https://github.com/Gaurav-Kumar98/CVInsight/wiki).
  - [Home](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Home)
  - [Installation and Setup](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Installation-and-Setup)
  - [User Guide](https://github.com/Gaurav-Kumar98/CVInsight/wiki/User-Guide)
  - [Plugin System](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Plugin-System)
  - [Technical Documentation](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Technical-Documentation)
  - [Examples and Tutorials](https://github.com/Gaurav-Kumar98/CVInsight/wiki/Examples-and-Tutorials)

- **Blog Post**: Learn more about the development process in the article [Building a Resume Parser with LLMs: A Step-by-Step Guide](https://www.linkedin.com/pulse/building-resume-parser-llms-step-by-step-guide-gaurav-kumar-82pqc)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.