# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from unittest.mock import Mock, patch

import pytest

from lighteval.models.endpoints.openai_model import OpenAIClient, OpenAIModelConfig
from lighteval.models.model_input import GenerationParameters


class TestOpenAIModelConfig:
    """Tests for OpenAIModelConfig configuration class."""

    @pytest.mark.parametrize(
        "config_path, expected_config",
        [
            (
                "examples/model_configs/openai_model.yaml",
                {
                    "model_name": "gpt-3.5-turbo",
                    "base_url": None,
                    "api_key": None,
                    "organization": None,
                    "concurrent_requests": 10,
                    "max_model_length": None,
                    "api_max_retry": 8,
                    "api_retry_sleep": 1.0,
                    "api_retry_multiplier": 2.0,
                    "timeout": None,
                    "system_prompt": None,
                    "generation_parameters": {
                        "block_size": None,
                        "num_blocks": None,
                        "cache_implementation": None,
                        "early_stopping": None,
                        "frequency_penalty": 0.0,
                        "length_penalty": None,
                        "max_new_tokens": 256,
                        "min_new_tokens": None,
                        "min_p": None,
                        "presence_penalty": None,
                        "repetition_penalty": None,
                        "seed": 42,
                        "stop_tokens": None,
                        "temperature": 0.7,
                        "top_k": None,
                        "top_p": 0.9,
                        "truncate_prompt": None,
                        "response_format": None,
                    },
                    "cache_dir": "~/.cache/huggingface/lighteval",
                    "tokenizer_type": "tiktoken",
                    "tokenizer_encoding": None,
                    "is_reasoning_model": False,
                    "supports_temperature": True,
                    "supports_stop_sequences": True,
                    "max_tokens_param_name": "max_completion_tokens",
                    "provider_name": "openai",
                    "provider": None,
                },
            ),
        ],
    )
    def test_from_path(self, config_path, expected_config):
        """Test loading configuration from YAML file."""
        config = OpenAIModelConfig.from_path(config_path)
        assert config.model_dump() == expected_config

    def test_config_creation_minimal(self):
        """Test creating config with minimal required parameters."""
        config = OpenAIModelConfig(model_name="gpt-4")
        assert config.model_name == "gpt-4"
        assert config.concurrent_requests == 10
        assert config.api_max_retry == 8
        assert config.api_retry_sleep == 1.0
        assert config.api_retry_multiplier == 2.0

    def test_config_creation_full(self):
        """Test creating config with all parameters."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="sk-test-key",
            base_url="https://api.openai.com/v1",
            organization="org-123",
            concurrent_requests=5,
            max_model_length=8192,
            api_max_retry=5,
            api_retry_sleep=2.0,
            api_retry_multiplier=1.5,
            timeout=30.0,
            generation_parameters=GenerationParameters(temperature=0.5, max_new_tokens=100),
            system_prompt="You are a helpful assistant.",
        )
        assert config.model_name == "gpt-4"
        assert config.api_key == "sk-test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.organization == "org-123"
        assert config.concurrent_requests == 5
        assert config.max_model_length == 8192
        assert config.api_max_retry == 5
        assert config.api_retry_sleep == 2.0
        assert config.api_retry_multiplier == 1.5
        assert config.timeout == 30.0
        assert config.generation_parameters.temperature == 0.5
        assert config.generation_parameters.max_new_tokens == 100
        assert config.system_prompt == "You are a helpful assistant."

    @pytest.mark.parametrize(
        "args_string, expected_values",
        [
            (
                "model_name=gpt-3.5-turbo,api_key=sk-test,concurrent_requests=5",
                {"model_name": "gpt-3.5-turbo", "api_key": "sk-test", "concurrent_requests": "5"},
            ),
            (
                "model_name=gpt-4,generation_parameters={temperature:0.7,max_new_tokens:100}",
                {
                    "model_name": "gpt-4",
                    "generation_parameters": {"temperature": 0.7, "max_new_tokens": 100},
                },
            ),
        ],
    )
    def test_from_args(self, args_string, expected_values):
        """Test parsing configuration from argument string."""
        config = OpenAIModelConfig.from_args(args_string)
        for key, value in expected_values.items():
            config_value = getattr(config, key)
            if key == "generation_parameters":
                assert config_value.temperature == value["temperature"]
                assert config_value.max_new_tokens == value["max_new_tokens"]
            else:
                assert str(config_value) == str(value)


class TestOpenAIClient:
    """Tests for OpenAIClient model client."""

    @pytest.fixture
    def mock_openai_client(self):
        """Fixture to mock OpenAI client."""
        with patch("lighteval.models.endpoints.openai_model.OpenAI") as mock:
            yield mock

    @pytest.fixture
    def mock_tiktoken(self):
        """Fixture to mock tiktoken."""
        with patch("lighteval.models.endpoints.openai_model.tiktoken") as mock:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoding.decode.return_value = "test"
            mock.encoding_for_model.return_value = mock_encoding
            yield mock

    @pytest.fixture
    def basic_config(self):
        """Fixture providing basic config."""
        return OpenAIModelConfig(
            model_name="gpt-3.5-turbo",
            generation_parameters=GenerationParameters(temperature=0.7, max_new_tokens=100),
        )

    def test_client_initialization(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test client initialization."""
        client = OpenAIClient(basic_config)

        assert client.model == "gpt-3.5-turbo"
        assert client.concurrent_requests == 10
        assert client.API_MAX_RETRY == 8
        assert client.API_RETRY_SLEEP == 1.0
        assert client.API_RETRY_MULTIPLIER == 2.0
        assert client.generation_parameters.temperature == 0.7

        # Verify OpenAI client was initialized
        mock_openai_client.assert_called_once()

    def test_client_initialization_with_custom_params(self, mock_openai_client, mock_tiktoken):
        """Test client initialization with custom parameters."""
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="sk-test",
            base_url="https://custom.api.com/v1",
            organization="org-123",
            timeout=30.0,
        )

        client = OpenAIClient(config)

        # Verify OpenAI client was initialized with correct params
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["base_url"] == "https://custom.api.com/v1"
        assert call_kwargs["organization"] == "org-123"
        assert call_kwargs["timeout"] == 30.0

    def test_tokenizer_property(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test tokenizer property."""
        client = OpenAIClient(basic_config)
        tokenizer = client.tokenizer

        # Test encode
        tokens = tokenizer.encode("test text")
        assert tokens == [1, 2, 3, 4, 5]

        # Test decode
        text = tokenizer.decode([1, 2, 3])
        assert text == "test"

    def test_add_special_tokens_property(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test add_special_tokens property."""
        client = OpenAIClient(basic_config)
        assert client.add_special_tokens is False

    def test_max_length_property_known_model(self, mock_openai_client, mock_tiktoken):
        """Test max_length property for known models."""
        test_cases = [
            ("gpt-3.5-turbo", 16385),
            ("gpt-4", 8192),
            ("gpt-4-turbo", 128000),
            ("gpt-4o", 128000),
            ("o1-preview", 128000),
            ("o1", 200000),
        ]

        for model_name, expected_length in test_cases:
            config = OpenAIModelConfig(model_name=model_name)
            client = OpenAIClient(config)
            assert client.max_length == expected_length

    def test_max_length_property_unknown_model(self, mock_openai_client, mock_tiktoken):
        """Test max_length property for unknown models."""
        config = OpenAIModelConfig(model_name="unknown-model")
        client = OpenAIClient(config)
        assert client.max_length == 4096  # Default

    def test_max_length_property_custom(self, mock_openai_client, mock_tiktoken):
        """Test max_length property with custom value."""
        config = OpenAIModelConfig(model_name="gpt-4", max_model_length=100000)
        client = OpenAIClient(config)
        assert client.max_length == 100000

    def test_is_reasoning_model(self, mock_openai_client, mock_tiktoken):
        """Test reasoning model detection."""
        # Reasoning model
        config = OpenAIModelConfig(model_name="o1", is_reasoning_model=True)
        client = OpenAIClient(config)
        assert client._is_reasoning_model() is True

        # Non-reasoning models
        for model_name in ["gpt-4", "gpt-3.5-turbo", "gpt-4o", "o1"]:
            config = OpenAIModelConfig(model_name=model_name, is_reasoning_model=False)
            client = OpenAIClient(config)
            assert client._is_reasoning_model() is False

    def test_prepare_stop_sequence(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test stop sequence preparation."""
        client = OpenAIClient(basic_config)

        # Valid sequences
        assert client._prepare_stop_sequence(["stop1", "stop2"]) == ["stop1", "stop2"]

        # Empty sequences filtered out
        assert client._prepare_stop_sequence(["stop1", "", "  ", "stop2"]) == ["stop1", "stop2"]

        # All empty
        assert client._prepare_stop_sequence(["", "  "]) is None

        # None input
        assert client._prepare_stop_sequence(None) is None

    def test_prepare_max_new_tokens(self, mock_openai_client, mock_tiktoken):
        """Test max new tokens preparation."""
        # Regular model
        config = OpenAIModelConfig(model_name="gpt-4")
        client = OpenAIClient(config)

        assert client._prepare_max_new_tokens(100) == 100
        assert client._prepare_max_new_tokens(0) is None
        assert client._prepare_max_new_tokens(-1) is None

        # Reasoning model (should multiply by 10)
        config_o1 = OpenAIModelConfig(model_name="o1-preview", is_reasoning_model=True)
        client_o1 = OpenAIClient(config_o1)

        assert client_o1._prepare_max_new_tokens(100) == 1000
        # Should cap at max_length
        assert client_o1._prepare_max_new_tokens(100000) == 128000

    def test_prepare_response_format(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test response format preparation."""
        client = OpenAIClient(basic_config)

        # No grammar
        assert client._prepare_response_format(None) is None

        # JSON grammar
        mock_grammar = Mock()
        mock_grammar.type = "json"
        mock_grammar.value = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = client._prepare_response_format(mock_grammar)
        assert result["type"] == "json_schema"
        assert "json_schema" in result
        assert result["json_schema"]["name"] == "response"
        assert result["json_schema"]["strict"] is True

    def test_loglikelihood_not_implemented(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test that loglikelihood raises NotImplementedError."""
        client = OpenAIClient(basic_config)

        # Create a mock Doc to pass to the method
        mock_doc = Mock()
        mock_doc.ctx = "context"
        mock_doc.cont = "continuation"

        with pytest.raises(NotImplementedError) as exc_info:
            # The actual implementation raises NotImplementedError
            # We need to call the underlying method, not through the cache
            client.loglikelihood.__wrapped__(client, [mock_doc])

        assert "log likelihoods" in str(exc_info.value).lower()

    def test_loglikelihood_rolling_not_implemented(self, mock_openai_client, mock_tiktoken, basic_config):
        """Test that loglikelihood_rolling raises NotImplementedError."""
        client = OpenAIClient(basic_config)

        # Create a mock Doc to pass to the method
        mock_doc = Mock()
        mock_doc.ctx = "context"

        with pytest.raises(NotImplementedError) as exc_info:
            # The actual implementation raises NotImplementedError
            # We need to call the underlying method, not through the cache
            client.loglikelihood_rolling.__wrapped__(client, [mock_doc])

        assert "rolling log likelihoods" in str(exc_info.value).lower()
