from dataclasses import dataclass

from mage_ai.data_preparation.models.constants import AIMode
from mage_ai.shared.config import BaseConfig


@dataclass
class OpenAIConfig(BaseConfig):
    openai_api_key: str = None
    openai_base_url: str = None
    openai_model: str = None


@dataclass
class OpenAICompatibleConfig(BaseConfig):
    """Configuration for any OpenAI-compatible provider (Ollama, Groq, Together AI, etc.)

    Set openai_base_url to the provider's API endpoint, e.g.:
      - Ollama (local):  http://localhost:11434/v1
      - Groq:            https://api.groq.com/openai/v1
      - Together AI:     https://api.together.xyz/v1
      - LM Studio:       http://localhost:1234/v1
    """
    openai_api_key: str = None
    openai_base_url: str = None
    openai_model: str = None


@dataclass
class HuggingFaceConfig(BaseConfig):
    huggingface_api: str = None
    huggingface_inference_api_token: str = None


@dataclass
class AIConfig(BaseConfig):
    mode: AIMode = AIMode.OPEN_AI
    open_ai_config: OpenAIConfig = OpenAIConfig
    open_ai_compatible_config: OpenAICompatibleConfig = OpenAICompatibleConfig
    hugging_face_config: HuggingFaceConfig = HuggingFaceConfig
