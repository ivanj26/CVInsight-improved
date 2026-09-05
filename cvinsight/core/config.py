import os
from dotenv import load_dotenv
from . import constants

# Load environment variables from .env file
load_dotenv()

# API Url
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL")

# Base URL of the OpenAI-compatible provider used for extraction
TOKENROUTER_API_URL = os.environ.get("TOKENROUTER_API_URL", "https://api.tokenrouter.com/v1")

# Local OpenCode server used for extraction when enabled.
OPENCODE_URL = os.environ.get("OPENCODE_URL", "http://localhost:4096").rstrip("/")
OPENCODE_ENABLED = os.environ.get("OPENCODE_ENABLED", "true").lower() == "true"
OPENCODE_PROVIDER_ID = os.environ.get("OPENCODE_PROVIDER_ID")
OPENCODE_MODEL_ID = os.environ.get("OPENCODE_MODEL_ID")

# API keys - get but don't raise error (handled by LLMService)
TOKENROUTER_API_KEY = os.environ.get("TOKENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# LLM Models
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "deepseek-v4-flash")

# LLM Model for Generative text (use for generate new contents)
DEFAULT_GEN_AI_LLM_MODEL = os.environ.get("DEFAULT_GEN_AI_LLM_MODEL", "deepseek-v4-flash")

# Date formats
DATE_FORMAT = constants.DEFAULT_DATE_FORMAT

# PDF processing configuration
PDF_EXTRACTION_METHOD = "PyPDF2"  # Options: 'PyPDF2', 'custom'

# File validation
MAX_PDF_SIZE_MB = constants.MAX_FILE_SIZE_MB  # Maximum PDF file size in MB
ALLOWED_FILE_EXTENSIONS = constants.RESUME_FILE_EXTENSIONS  # Allowed file extensions

# Directory configuration
RESUME_DIR = os.environ.get("RESUME_DIR", "./Resumes")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./Results")

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = os.environ.get("LOG_FILE", "cvinsight.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
TOKEN_LOG_RETENTION_DAYS = int(os.environ.get("TOKEN_LOG_RETENTION_DAYS", "7"))  # Keep token logs for 7 days by default
LOG_MAX_SIZE_MB = int(os.environ.get("LOG_MAX_SIZE_MB", "5"))  # Maximum log file size in MB
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "3"))  # Number of backup log files to keep

# Debug mode
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Plugin system configuration
PLUGINS_DIR = os.environ.get("PLUGINS_DIR", "./plugins")
CUSTOM_PLUGINS_DIR = os.environ.get("CUSTOM_PLUGINS_DIR", "./custom_plugins")
ENABLE_CUSTOM_PLUGINS = True

# LLM configuration for plugins
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", str(constants.DEFAULT_MODEL_TEMPERATURE)))

# LLM request limits
# Worst-case wall clock per extraction is roughly LLM_REQUEST_TIMEOUT * (LLM_MAX_RETRIES + 1)
LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))  # Seconds before an LLM request is aborted
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))  # Retries attempted by the OpenAI client on failure
