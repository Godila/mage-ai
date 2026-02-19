import json
import os
from typing import Dict

import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI as OpenAILib

from mage_ai.ai.ai_client import AIClient
from mage_ai.data_cleaner.transformer_actions.constants import ActionType, Axis
from mage_ai.data_preparation.models.constants import (
    BlockLanguage,
    BlockType,
    PipelineType,
)
from mage_ai.data_preparation.repo_manager import get_repo_config
from mage_ai.io.base import DataSource
from mage_ai.orchestration.ai.config import OpenAIConfig

CLASSIFICATION_FUNCTION_NAME = "classify_description"
tools = [
    {
        "type": "function",
        "function": {
            "name": CLASSIFICATION_FUNCTION_NAME,
            "description": "Classify the code description provided into following properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    BlockType.__name__: {
                        "type": "string",
                        "description": "Type of the code block. It either "
                                       "loads data from a source, export data to a source "
                                       "or transform data from one format to another.",
                        "enum": [f"{BlockType.__name__}__data_exporter",
                                 f"{BlockType.__name__}__data_loader",
                                 f"{BlockType.__name__}__transformer"]
                    },
                    BlockLanguage.__name__: {
                        "type": "string",
                        "description": "Programming language of the code block. "
                                       f"Default value is {BlockLanguage.__name__}__python.",
                        "enum": [
                            f"{BlockLanguage.__name__}__{type.name.lower()}"
                            for type in BlockLanguage]
                    },
                    PipelineType.__name__: {
                        "type": "string",
                        "description": "Type of pipeline to build. Default value is "
                                       f"{PipelineType.__name__}__python if pipeline type "
                                       "is not mentioned in the description.",
                        "enum": [
                            f"{PipelineType.__name__}__{type.name.lower()}"
                            for type in PipelineType]
                    },
                    ActionType.__name__: {
                        "type": "string",
                        "description": f"If {BlockType.__name__} is transformer, "
                                       f"{ActionType.__name__} specifies what kind "
                                       "of action the code performs.",
                        "enum": [f"{ActionType.__name__}__{type.name.lower()}"
                                 for type in ActionType]
                    },
                    DataSource.__name__: {
                        "type": "string",
                        "description": f"If {BlockType.__name__} is data_loader or "
                                       f"data_exporter, {DataSource.__name__} field specify "
                                       "where the data loads from or exports to.",
                        "enum": [f"{DataSource.__name__}__{type.name.lower()}"
                                 for type in DataSource]
                    },
                },
                "required": [BlockType.__name__, BlockLanguage.__name__, PipelineType.__name__],
            },
        }
    },
]
GPT_MODEL_DEFAULT = "gpt-4o"


class OpenAIClient(AIClient):
    def __init__(self, open_ai_config: OpenAIConfig):
        repo_config = get_repo_config()
        openai_api_key = (
            repo_config.openai_api_key
            or open_ai_config.openai_api_key
            or os.getenv('OPENAI_API_KEY')
        )
        base_url = (
            open_ai_config.openai_base_url
            or os.getenv('OPENAI_BASE_URL')
            or None
        )
        self.model = (
            open_ai_config.openai_model
            or os.getenv('OPENAI_MODEL')
            or GPT_MODEL_DEFAULT
        )

        openai.api_key = openai_api_key

        # LangChain ChatOpenAI wrapper — uses /v1/chat/completions (supported by all providers)
        # The legacy langchain.llms.OpenAI used /v1/completions which most compatible
        # providers (MiniMax, Groq, Together AI, etc.) do NOT support → 404 error.
        llm_kwargs = dict(
            openai_api_key=openai_api_key,
            model_name=self.model,
            temperature=0,
        )
        if base_url:
            llm_kwargs['openai_api_base'] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)

        # Official OpenAI SDK — supports base_url for OpenAI-compatible providers
        client_kwargs = dict(api_key=openai_api_key)
        if base_url:
            client_kwargs['base_url'] = base_url
        self.openai_client = OpenAILib(**client_kwargs)

    @staticmethod
    def __strip_thinking_tags(text: str) -> str:
        """Remove <think>...</think> reasoning blocks emitted by some models (e.g. MiniMax)."""
        import re
        # Remove all <think>…</think> sections (greedy=False so we handle multiple blocks)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def __extract_last_json(text: str) -> dict:
        """Return the *last* valid JSON object found in text.

        Models sometimes emit an example JSON before the real answer.
        We scan from the rightmost '}' backwards to find the outermost
        valid object that belongs to the actual answer.
        """
        end = len(text) - 1
        while end >= 0:
            end = text.rfind('}', 0, end + 1)
            if end == -1:
                break
            # Walk left to find the matching '{'
            depth = 0
            for i in range(end, -1, -1):
                if text[i] == '}':
                    depth += 1
                elif text[i] == '{':
                    depth -= 1
                if depth == 0:
                    candidate = text[i:end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    break
            end -= 1  # try the next '}' to the left
        raise json.JSONDecodeError("No valid JSON object found", text, 0)

    @staticmethod
    def __classification_fallback_messages(messages):
        """Replace the last user message with a structured JSON-only classification request.

        The original block_description is preserved as context so the model
        knows what to classify. The example is omitted to avoid the model
        echoing it back as its answer.
        """
        # Extract the original block description from the first user message
        original_content = next(
            (m['content'] for m in messages if m.get('role') == 'user'), ''
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a code block classifier. "
                    "You MUST respond with ONLY a single JSON object — "
                    "no thinking, no explanation, no markdown fences. "
                    'Keys: "BlockType" ("data_loader"|"data_exporter"|"transformer"), '
                    '"BlockLanguage" ("python"|"sql"|"r"|"yaml"|"markdown"), '
                    '"PipelineType" ("python"|"integration"|"streaming").'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Classify this block description:\n{original_content}\n\n"
                    "Reply with ONLY the JSON object, nothing else."
                ),
            },
        ]

    async def __chat_completion_request(self, messages):
        """Call chat completions API with tool use.

        Handles two failure modes:
        1. Provider raises an error on the tools parameter → retry without tools.
        2. Provider accepts tools but returns content instead of tool_calls
           (tool_calls is None) → this is handled downstream in find_block_params.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": CLASSIFICATION_FUNCTION_NAME},
                },
            )
            # If provider accepted the call but ignored tool_choice and put
            # the answer in content, retry with an explicit JSON instruction
            # so __parse_block_params_from_content can handle it.
            msg = response.choices[0].message
            if not msg.tool_calls and msg.content:
                print("[WARN] Provider returned content instead of tool_calls. "
                      "Retrying with explicit JSON prompt.")
                fallback_resp = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=self.__classification_fallback_messages(messages),
                )
                return fallback_resp
            return response
        except Exception as e:
            print(f"[WARN] Tool-calling request failed ({e}). Retrying without tools.")
            try:
                return self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=self.__classification_fallback_messages(messages),
                )
            except Exception as e2:
                print(f"Unable to generate ChatCompletion response: {e2}")
                return e2

    async def inference_with_prompt(
            self,
            variable_values: Dict[str, str],
            prompt_template: str,
            is_json_response: bool = True
    ):
        """Generic function to call OpenAI LLM and return JSON response by default.

        Fill variables and values into template, and run against LLM
        to genenrate JSON format response.

        Args:
            variable_values: all required variable and values in prompt.
            prompt_template: prompt template for LLM call.
            is_json_response: default is json formatted response.

        Returns:
            We typically suggest response in JSON format. For example:
                {
                    'action_code': 'grade == 5 or grade == 6',
                    'arguments': ['class']
                }
        """
        filled_prompt = PromptTemplate(
            input_variables=list(variable_values.keys()),
            template=prompt_template,
        )
        chain = LLMChain(llm=self.llm, prompt=filled_prompt)
        if is_json_response:
            resp = await chain.arun(variable_values)
            # Strip <think>...</think> reasoning blocks emitted by some models
            resp = self.__strip_thinking_tags(resp)
            # Try to extract the last valid JSON object from the response
            try:
                return self.__extract_last_json(resp)
            except json.JSONDecodeError:
                pass
            # Fallback: wrap bare key:value pairs in braces
            if not resp.startswith('{') or not resp.endswith('}'):
                resp = f'{{{resp.strip()}}}'
            if resp:
                try:
                    return json.loads(resp)
                except json.decoder.JSONDecodeError as err:
                    print(f'[ERROR] OpenAIClient.inference_with_prompt {resp}: {err}.')
                    return resp
            else:
                return {}
        resp = await chain.arun(variable_values)
        return self.__strip_thinking_tags(resp)

    def __parse_argument_value(self, value: str) -> str:
        if value is None:
            return None
        # If model returned value does not contain '__' as we suggested in the tools
        # then return the value as it is.
        if '__' not in value:
            return value
        return value.lower().split('__')[1]

    def __load_template_params(self, function_args: json):
        block_type = BlockType(self.__parse_argument_value(function_args[BlockType.__name__]))
        block_language = BlockLanguage(
                            self.__parse_argument_value(
                                function_args.get(BlockLanguage.__name__)
                            ) or "python")
        pipeline_type = PipelineType(
                            self.__parse_argument_value(
                                function_args.get(PipelineType.__name__)
                            ) or "python")
        config = {}
        config['action_type'] = self.__parse_argument_value(
                                    function_args.get(ActionType.__name__))
        if config['action_type']:
            if config['action_type'] in [
                ActionType.FILTER,
                ActionType.DROP_DUPLICATE,
                ActionType.REMOVE,
                ActionType.SORT
            ]:
                config['axis'] = Axis.ROW
            else:
                config['axis'] = Axis.COLUMN
        config['data_source'] = self.__parse_argument_value(
                                    function_args.get(DataSource.__name__))
        return block_type, block_language, pipeline_type, config

    def __parse_block_params_from_content(self, content: str):
        """Fallback parser: extract block classification from plain-text JSON content.

        Used when the provider does not support function calling and returns
        a JSON object in message.content instead of a tool_call.

        Handles:
        - <think>...</think> reasoning blocks (MiniMax, DeepSeek-R1, etc.)
        - Markdown code fences
        - Multiple JSON objects in the text (picks the last valid one)
        """
        # 1. Remove reasoning/thinking blocks
        text = self.__strip_thinking_tags(content)

        # 2. Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop the opening fence line and the closing fence line (if present)
            inner = lines[1:]
            if inner and inner[-1].strip().startswith("```"):
                inner = inner[:-1]
            text = "\n".join(inner).strip()

        # 3. Extract the last valid JSON object (model may echo an example first)
        try:
            raw = self.__extract_last_json(text)
        except json.JSONDecodeError as e:
            raise Exception(
                f"Could not parse block classification from model content: {e}\n"
                f"Raw content: {content!r}"
            )

        # Normalise keys — model may return e.g. "block_type" or "BlockType"
        def _get(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        bt_raw = _get(raw, 'BlockType', 'block_type', 'type')
        bl_raw = _get(raw, 'BlockLanguage', 'block_language', 'language')
        pt_raw = _get(raw, 'PipelineType', 'pipeline_type', 'pipeline')

        # Accept both bare values ("python") and namespaced ("BlockLanguage__python")
        def _parse(val, default):
            if val is None:
                return default
            v = str(val).lower()
            return v.split('__')[-1] if '__' in v else v

        bt_val = _parse(bt_raw, 'transformer')
        bl_val = _parse(bl_raw, 'python')
        pt_val = _parse(pt_raw, 'python')

        # Build a synthetic function_args dict that __load_template_params understands
        function_args = {
            'BlockType': f'BlockType__{bt_val}',
            'BlockLanguage': f'BlockLanguage__{bl_val}',
            'PipelineType': f'PipelineType__{pt_val}',
        }
        return self.__load_template_params(function_args)

    async def find_block_params(
            self,
            block_description: str):
        messages = [{'role': 'user', 'content': block_description}]
        # Fetch response from API with retries
        max_retries = 2
        attempt = 0
        response = await self.__chat_completion_request(messages)
        while attempt < max_retries and isinstance(response, Exception):
            response = await self.__chat_completion_request(messages)
            attempt += 1
        if isinstance(response, Exception):
            raise Exception("Error in __chat_completion_request after retries: " + str(response))

        message = response.choices[0].message

        # --- Path 1: provider returned a proper tool call (function calling) ---
        if message.tool_calls:
            arguments = message.tool_calls[0].function.arguments
            if arguments:
                function_args = json.loads(arguments)
                block_type, block_language, pipeline_type, config = \
                    self.__load_template_params(function_args)
                return dict(
                    block_type=block_type,
                    block_language=block_language,
                    pipeline_type=pipeline_type,
                    config=config,
                )
            raise Exception('Tool call returned empty arguments.')

        # --- Path 2: no tool_calls — try to parse JSON from message content ---
        content = message.content or ""
        if content.strip():
            block_type, block_language, pipeline_type, config = \
                self.__parse_block_params_from_content(content)
            return dict(
                block_type=block_type,
                block_language=block_language,
                pipeline_type=pipeline_type,
                config=config,
            )

        raise Exception(
            'Failed to interpret the description as a block template: '
            'no tool_calls and no content in model response.'
        )
