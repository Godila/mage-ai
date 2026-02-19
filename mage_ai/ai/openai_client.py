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
    def __classification_fallback_messages(messages):
        """Append a plain-JSON instruction for providers that ignore tool_choice."""
        return messages + [{
            "role": "user",
            "content": (
                "Respond ONLY with a JSON object (no markdown, no explanation) "
                "with exactly these keys:\n"
                '  "BlockType": one of "data_loader", "data_exporter", "transformer"\n'
                '  "BlockLanguage": one of "python", "sql", "r", "yaml", "markdown"\n'
                '  "PipelineType": one of "python", "integration", "streaming"\n'
                "Example: "
                '{"BlockType": "data_loader", "BlockLanguage": "python", "PipelineType": "python"}'
            ),
        }]

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
            # If the model response didn't start with
            # '{' and end with '}' follwing in the JSON format,
            # then we will add '{' and '}' to make it JSON format.
            if not resp.startswith('{') and not resp.endswith('}'):
                resp = f'{{{resp.strip()}}}'
            if resp:
                try:
                    return json.loads(resp)
                except json.decoder.JSONDecodeError as err:
                    print(f'[ERROR] OpenAIClient.inference_with_prompt {resp}: {err}.')
                    return resp
            else:
                return {}
        return await chain.arun(variable_values)

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
        """
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Try to find a JSON object anywhere in the response
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]

        try:
            raw = json.loads(text)
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
