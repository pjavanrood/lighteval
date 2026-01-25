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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)

# Check for OpenAI and tiktoken availability
if is_package_available("openai"):
    from openai import OpenAI, BadRequestError, APIError
else:
    OpenAI = Mock()
    BadRequestError = Mock()
    APIError = Mock()

if is_package_available("tiktoken"):
    import tiktoken
else:
    tiktoken = Mock()


class OpenAIModelConfig(ModelConfig):
    """Configuration class for OpenAI API client.

    This configuration is used to connect to OpenAI's API for language model inference.
    It supports all OpenAI models including GPT-4, GPT-3.5-Turbo, and o1 series.

    OpenAI API doc: https://platform.openai.com/docs/api-reference/introduction

    Attributes:
        model_name (str):
            OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo", "gpt-4o").
        api_key (str | None):
            OpenAI API key for authentication. If None, reads from OPENAI_API_KEY environment variable.
        base_url (str | None):
            Custom base URL for the API. If None, uses OpenAI's default URL (https://api.openai.com/v1).
            Useful for using custom endpoints or compatible services.
        organization (str | None):
            OpenAI organization ID. If None, uses default organization from account.
        concurrent_requests (int):
            Maximum number of concurrent API requests to execute in parallel.
            Higher values can improve throughput but may hit rate limits. Default is 10.
        max_model_length (int | None):
            Maximum context length for the model. If None, infers from model name using tiktoken.
        api_max_retry (int):
            Maximum number of retries for API requests. Default is 8.
        api_retry_sleep (float):
            Initial sleep time (in seconds) between retries. Default is 1.0.
        api_retry_multiplier (float):
            Multiplier for increasing sleep time between retries. Default is 2.0.
        timeout (float | None):
            Request timeout in seconds. Default is None (no timeout).
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None):
            Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"):
            Directory to cache the model.

    Example:
        ```python
        config = OpenAIModelConfig(
            model_name="gpt-4",
            api_key="sk-...",
            concurrent_requests=5,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    concurrent_requests: int = 10
    max_model_length: int | None = None

    api_max_retry: int = 8
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0
    timeout: float | None = None


@requires("openai", "tiktoken")
class OpenAIClient(LightevalModel):
    """OpenAI API client for lighteval.

    This client uses the official OpenAI Python SDK to interface with OpenAI's API.
    It supports text generation, handles retries, and integrates with lighteval's caching system.
    """

    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: OpenAIModelConfig) -> None:
        """Initialize OpenAI client.

        IMPORTANT: Your API key should be set in the OPENAI_API_KEY environment variable
        or passed explicitly in the config.
        """
        self.config = config
        self.model = config.model_name
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self._max_length = config.max_model_length

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.timeout = config.timeout

        # Initialize OpenAI client
        client_kwargs = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.organization:
            client_kwargs["organization"] = config.organization
        if config.timeout:
            client_kwargs["timeout"] = config.timeout

        self.client = OpenAI(**client_kwargs)

        # Initialize tokenizer using tiktoken
        try:
            self._tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            logger.warning(f"Model {self.model} not found in tiktoken, using cl100k_base encoding")
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        self.pairwise_tokenization = False
        self.prompt_manager = PromptManager(
            use_chat_template=True, tokenizer=self.tokenizer, system_prompt=config.system_prompt
        )

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    def _is_reasoning_model(self) -> bool:
        """Check if the model is a reasoning model (o1 series)."""
        return "o1" in self.model.lower()

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if stop_sequence:
            # Filter out empty or whitespace-only stop sequences
            stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence if stop_sequence else None

    def _prepare_max_new_tokens(self, max_new_tokens) -> int | None:
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if self._is_reasoning_model():
            # Reasoning models need more tokens for reasoning content
            max_new_tokens = min(max_new_tokens * 10, self.max_length)
            logger.warning(
                f"Reasoning model detected, increasing max_new_tokens to {max_new_tokens} to allow for reasoning tokens"
            )

        return max_new_tokens

    def _prepare_response_format(self, grammar):
        """Convert grammar to response_format for OpenAI API."""
        if not grammar:
            return None

        if hasattr(grammar, "type") and grammar.type == "json" and hasattr(grammar, "value"):
            schema = grammar.value.copy()
            if schema.get("type") == "json_schema":
                schema.pop("type")
            if "type" not in schema:
                schema["type"] = "object"

            return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": True}}

        return None

    def __call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence, grammar):  # noqa: C901
        """Make API call with retries."""
        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)
        response_format = self._prepare_response_format(grammar)

        # Prepare kwargs for chat completion
        kwargs = {
            "model": self.model,
            "messages": prompt,
            "n": num_samples,
        }

        # Add optional parameters
        if response_format:
            kwargs["response_format"] = response_format
        if return_logits:
            kwargs["logprobs"] = True
        if stop_sequence:
            kwargs["stop"] = stop_sequence

        # O1 models don't support temperature, top_p, or stop sequences
        if self._is_reasoning_model():
            logger.warning("O1 models do not support temperature, top_p, or stop sequences. Disabling.")
            kwargs.pop("stop", None)
        else:
            # Add generation parameters
            gen_params = self.generation_parameters.to_litellm_dict()
            # Map parameters to OpenAI's naming
            if "max_completion_tokens" in gen_params:
                gen_params.pop("max_completion_tokens")  # We'll set it separately
            if "stop" in gen_params:
                gen_params.pop("stop")  # Already handled above
            kwargs.update(gen_params)

        # Set max tokens - use max_completion_tokens for OpenAI
        if max_new_tokens:
            kwargs["max_completion_tokens"] = max_new_tokens

        errors = []
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # If response is empty, retry
                if not content and attempt < self.API_MAX_RETRY - 1:
                    logger.info("Response is empty, retrying")
                    wait_time = min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                    time.sleep(wait_time)
                    continue

                return response

            except BadRequestError as e:
                errors.append(e)
                error_message = str(e)
                # Check for content filtering
                if "content management policy" in error_message.lower() or "content_filter" in error_message.lower():
                    logger.warning(f"Response was filtered due to content policy. Returning empty response.")
                    # Return a mock empty response
                    return type('obj', (object,), {
                        'choices': [type('obj', (object,), {
                            'message': type('obj', (object,), {'content': '', 'reasoning_content': None})()
                        })()]
                    })()
                # For other bad requests, don't retry
                logger.error(f"BadRequestError: {e}")
                break

            except Exception as e:
                errors.append(e)
                wait_time = min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                logger.warning(
                    f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response. Errors: {errors}")
        # Return a mock empty response
        return type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {'content': '', 'reasoning_content': None})()
            })()]
        })()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
        grammar=None,
    ):
        """Execute multiple API calls in parallel."""
        results = []

        # Convert single values to lists
        return_logitss = [return_logits for _ in prompts] if not isinstance(return_logits, list) else return_logits
        max_new_tokenss = [max_new_tokens for _ in prompts] if not isinstance(max_new_tokens, list) else max_new_tokens
        num_sampless = [num_samples for _ in prompts] if not isinstance(num_samples, list) else num_samples
        stop_sequencess = [stop_sequence for _ in prompts]
        grammars = [grammar for _ in prompts]

        assert (
            len(prompts) == len(return_logitss) == len(max_new_tokenss) == len(num_sampless) == len(stop_sequencess)
        ), (
            f"Length mismatch: prompts={len(prompts)}, return_logits={len(return_logitss)}, "
            f"max_new_tokens={len(max_new_tokenss)}, num_samples={len(num_sampless)}, stop_sequences={len(stop_sequencess)}"
        )

        with ThreadPoolExecutor(self.concurrent_requests) as executor:
            for entry in tqdm(
                executor.map(
                    self.__call_api,
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                    grammars,
                ),
                total=len(prompts),
            ):
                results.append(entry)

        if None in results:
            raise ValueError("Some entries are not annotated due to errors in API calls, please inspect and retry.")

        return results

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in dataset]
            max_new_tokens = split[0].generation_size
            return_logits = split[0].use_logits
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences
            grammar = split[0].generation_grammar

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(
                contexts, return_logits, max_new_tokens, num_samples, stop_sequence, grammar
            )

            for response, context in zip(responses, contexts):
                result: list[str] = [choice.message.content for choice in response.choices]
                reasonings: list[str | None] = [
                    getattr(choice.message, "reasoning_content", None) for choice in response.choices
                ]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result[0] else [""],
                    reasonings=reasonings,
                    input=context,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        """Return the tiktoken tokenizer."""
        # Create a mock tokenizer object that has the encode method
        class TiktokenWrapper:
            def __init__(self, encoding):
                self._encoding = encoding
                self.eos_token_id = None  # OpenAI handles this internally

            def encode(self, text, add_special_tokens=False):
                # tiktoken doesn't have add_special_tokens, it's handled by the API
                return self._encoding.encode(text)

            def decode(self, tokens):
                return self._encoding.decode(tokens)

            def batch_decode(self, token_lists, skip_special_tokens=True):
                return [self._encoding.decode(tokens) for tokens in token_lists]

        return TiktokenWrapper(self._tokenizer)

    @property
    def add_special_tokens(self) -> bool:
        """OpenAI API handles special tokens internally."""
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        if self._max_length is not None:
            return self._max_length

        # Model context lengths from OpenAI documentation
        # https://platform.openai.com/docs/models
        model_max_lengths = {
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0613": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-2024-04-09": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-4-1106-preview": 128000,
            "gpt-4-0125-preview": 128000,
            "gpt-4o": 128000,
            "gpt-4o-2024-05-13": 128000,
            "gpt-4o-2024-08-06": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4o-mini-2024-07-18": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-0125": 16385,
            "gpt-3.5-turbo-1106": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "o1-preview": 128000,
            "o1-preview-2024-09-12": 128000,
            "o1-mini": 128000,
            "o1-mini-2024-09-12": 128000,
            "o1": 200000,
        }

        max_length = model_max_lengths.get(self.model, self._DEFAULT_MAX_LENGTH)

        if self.model not in model_max_lengths:
            logger.warning(
                f"Model {self.model} not found in known models, using default max length {self._DEFAULT_MAX_LENGTH}"
            )

        # Cache the result
        self._max_length = max_length
        return max_length

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Tokenize the context and continuation and compute the log likelihood of those
        tokenized sequences.

        Note: OpenAI's API does not directly support computing log likelihoods for arbitrary
        continuations, so this method is not implemented.
        """
        raise NotImplementedError(
            "OpenAI API does not support computing log likelihoods for arbitrary continuations. "
            "Use generative tasks instead."
        )

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics.

        Note: OpenAI's API does not support this functionality directly.
        """
        raise NotImplementedError(
            "OpenAI API does not support computing rolling log likelihoods. "
            "Use generative tasks instead."
        )
