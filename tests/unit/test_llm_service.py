"""Unit tests for LLM service functionality."""
import pytest
from unittest.mock import MagicMock, patch
from cvinsight.core.llm_service import LLMService
from pydantic import BaseModel
from typing import List

class TestModel(BaseModel):
    """Test Pydantic model."""
    name: str
    skills: List[str]

@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM."""
    monkeypatch.setenv("TOKENROUTER_API_KEY", "test-tokenrouter-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr("cvinsight.core.llm_service.config.TOKENROUTER_API_KEY", "test-tokenrouter-key")
    monkeypatch.setattr("cvinsight.core.llm_service.config.DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch('cvinsight.core.llm_service.ChatOpenAI') as mock:
        mock_instance = MagicMock()
        mock_generation = MagicMock()
        mock_generation.text = '{"name": "John Doe", "skills": ["Python", "Java"]}'
        mock_generation.content = '{"name": "John Doe", "skills": ["Python", "Java"]}'
        
        mock_instance.invoke.return_value = mock_generation
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def llm_service(mock_llm):
    """Create LLM service instance."""
    return LLMService()

def test_llm_service_initialization(llm_service):
    """Test LLM service initialization."""
    assert llm_service is not None
    assert llm_service.model_name is not None
    assert llm_service.llm is not None

def test_create_extraction_chain(llm_service):
    """Test creating extraction chain."""
    chain = llm_service.create_extraction_chain(
        TestModel,
        "Extract information from: {text}",
        ["text"]
    )
    assert chain is not None

def test_extract_with_llm(llm_service, mock_llm):
    """Test extracting information with LLM."""
    # Set up a direct return value for the chain
    from unittest.mock import patch
    
    with patch.object(llm_service, 'create_extraction_chain') as mock_chain_creator:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"name": "John Doe", "skills": ["Python", "Java"]}
        mock_chain_creator.return_value = mock_chain
        
        prompt_template = "Extract information from: {text}"
        input_variables = ["text"]
        input_data = {"text": "John Doe is skilled in Python and Java"}
        
        result, token_usage = llm_service.extract_with_llm(
            TestModel,
            prompt_template,
            input_variables,
            input_data
        )
        
        assert isinstance(result, dict)
        assert "name" in result
        assert "skills" in result
        assert result["name"] == "John Doe"
        assert result["skills"] == ["Python", "Java"]
        assert isinstance(token_usage, dict)
        assert "total_tokens" in token_usage
        assert "prompt_tokens" in token_usage
        assert "completion_tokens" in token_usage

def test_extract_with_llm_error(llm_service, mock_llm):
    """Test extracting information with LLM error."""
    mock_llm.invoke.side_effect = Exception("API Error")
    
    prompt_template = "Extract information from: {text}"
    input_variables = ["text"]
    input_data = {"text": "test text"}
    
    result, token_usage = llm_service.extract_with_llm(
        TestModel,
        prompt_template,
        input_variables,
        input_data
    )
    
    assert isinstance(result, dict)
    assert not result
    assert token_usage["source"] == "error"
    assert token_usage["total_tokens"] == 0
    assert token_usage["prompt_tokens"] == 0
    assert token_usage["completion_tokens"] == 0

def test_extract_with_llm_empty_response(llm_service, mock_llm):
    """Test extracting information with empty response."""
    mock_llm.invoke.return_value = MagicMock(content="{}")
    
    prompt_template = "Extract information from: {text}"
    input_variables = ["text"]
    input_data = {"text": "test text"}
    
    result, token_usage = llm_service.extract_with_llm(
        TestModel,
        prompt_template,
        input_variables,
        input_data
    )
    
    assert isinstance(result, dict)
    assert not result.get("name")  # should not exist or be empty
    assert not result.get("skills")  # should not exist or be empty
    assert isinstance(token_usage, dict)
    assert "total_tokens" in token_usage
    assert "prompt_tokens" in token_usage
    assert "completion_tokens" in token_usage 