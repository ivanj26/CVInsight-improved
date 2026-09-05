"""Unit tests for LLM service functionality."""
import pytest
from unittest.mock import MagicMock, patch
from cvinsight.core.llm_service import LLMService
from cvinsight.core import config as llm_config
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
    monkeypatch.setattr(llm_config, "TOKENROUTER_API_KEY", "test-tokenrouter-key", raising=False)
    monkeypatch.setattr(llm_config, "DEEPSEEK_API_KEY", "test-deepseek-key", raising=False)
    monkeypatch.setattr(llm_config, "OPENCODE_ENABLED", False, raising=False)
    monkeypatch.setattr(llm_config, "OPENCODE_PROVIDER_ID", None, raising=False)
    monkeypatch.setattr(llm_config, "OPENCODE_MODEL_ID", None, raising=False)

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


def test_extract_with_opencode(llm_service, monkeypatch):
    """Test session creation, prompt delivery, response parsing, and cleanup."""
    session_response = MagicMock()
    session_response.json.return_value = {"id": "ses_test"}
    message_response = MagicMock()
    message_response.json.return_value = {
        "info": {"tokens": {"input": 10, "output": 4}},
        "parts": [{"type": "text", "text": '{"name":"Jane Doe","skills":["Go"]}'}],
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = [session_response, message_response]
    monkeypatch.setattr("cvinsight.core.llm_service.httpx.Client", MagicMock(return_value=client))
    monkeypatch.setattr(llm_config, "OPENCODE_PROVIDER_ID", "agentrouter")
    monkeypatch.setattr(llm_config, "OPENCODE_MODEL_ID", "deepseek-v4-flash")

    result, token_usage = llm_service._extract_with_opencode(
        TestModel,
        "Extract information from: {text}\n{format_instructions}",
        ["text"],
        {"text": "Jane Doe knows Go"},
    )

    assert result == {"name": "Jane Doe", "skills": ["Go"]}
    assert token_usage["total_tokens"] == 14
    assert client.post.call_args_list[1].args[0] == "http://localhost:4096/session/ses_test/message"
    assert client.post.call_args_list[1].kwargs["json"]["model"] == {
        "providerID": "agentrouter",
        "modelID": "deepseek-v4-flash",
    }
    client.delete.assert_called_once_with("http://localhost:4096/session/ses_test")


def test_generate_content_with_opencode(llm_service, monkeypatch):
    """Test generation keeps the response shape expected by recommendation plugins."""
    session_response = MagicMock()
    session_response.json.return_value = {"id": "ses_generate"}
    message_response = MagicMock()
    message_response.json.return_value = {
        "info": {"tokens": {"input": 12, "output": 8, "total": 20}},
        "parts": [{"type": "text", "text": '{"recommendations":["Use Go"]}'}],
    }
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.side_effect = [session_response, message_response]
    monkeypatch.setattr("cvinsight.core.llm_service.httpx.Client", MagicMock(return_value=client))
    monkeypatch.setattr(llm_config, "OPENCODE_PROVIDER_ID", "agentrouter")
    monkeypatch.setattr(llm_config, "OPENCODE_MODEL_ID", "deepseek-v4-flash")
    monkeypatch.setattr(llm_config, "OPENCODE_ENABLED", True)

    result, token_usage = llm_service.generate_content(
        [
            type("Message", (), {"role": "system", "content": "Return JSON"})(),
            type("Message", (), {"role": "user", "content": "Recommend Go"})(),
        ]
    )

    assert result.choices[0].message.content == '{"recommendations":["Use Go"]}'
    assert token_usage["total_tokens"] == 20
    assert client.post.call_args_list[1].kwargs["json"]["model"] == {
        "providerID": "agentrouter",
        "modelID": "deepseek-v4-flash",
    }
    client.delete.assert_called_once_with("http://localhost:4096/session/ses_generate")
