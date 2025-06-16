import os
from importlib import import_module
from typing import Dict, Type

import transformers
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))


class ModelRegistry:
    MODEL_MAPPINGS = {
        "gpt-4o-2024-11-20": ("openai_chat", "OpenAIChat"),
        "gpt-4-turbo": ("openai_chat", "OpenAIChat"),
        "gemini-2.0-flash": ("openai_chat", "OpenAIChat"),
        "gemini-2.0-pro-exp-02-05": ("openai_chat", "OpenAIChat"),
        "claude-3-7-sonnet-20250219": ("openai_chat", "OpenAIChat"),
        "llava-hf/llava-v1.6-mistral-7b-hf": ("llava_next_chat", "LlavaNextChat"),
        "lmms-lab/llava-onevision-qwen2-72b-ov-sft": ("llava_ov_chat", "LlavaOVChat"),
        "lmms-lab/llava-onevision-qwen2-72b-ov-chat": ("llava_ov_chat", "LlavaOVChat"),
        "microsoft/Magma-8B": ("magma_chat", "MagmaChat"),
        "Qwen/Qwen2.5-VL-7B-Instruct": ("qwen_chat", "QwenChat"),
        "deepseek-ai/deepseek-vl2": ("deepseek_chat", "DeepSeekVL2Chat"),
        "openbmb/MiniCPM-o-2_6": ("minicpm_chat", "MiniCPMOChat"),
        "mistral-community/pixtral-12b": ("pixtral_chat", "PixtralChat"),
        "microsoft/Phi-4-multimodal-instruct": ("phi_chat", "PhiChat"),
        "OpenGVLab/InternVL2-8B": ("internvl2_chat", "InternVL2Chat"),
    }

    _registered_models: Dict[str, Type] = {}

    @classmethod
    def register(cls, model_name: str) -> Type:
        if model_name in cls._registered_models:
            return cls._registered_models[model_name]

        if str(transformers.__version__) == "4.44.2":
            if model_name == "openbmb/MiniCPM-o-2_6":
                from .minicpm_chat import MiniCPMOChat

                cls._registered_models[model_name] = MiniCPMOChat
                return MiniCPMOChat
            raise ValueError(f"Model {model_name} not supported in transformers 4.44.2")

        if model_name not in cls.MODEL_MAPPINGS:
            raise ValueError(f"Unknown model: {model_name}")

        module_name, class_name = cls.MODEL_MAPPINGS[model_name]
        module = import_module(f"src.models.{module_name}")
        model_class = getattr(module, class_name)
        cls._registered_models[model_name] = model_class

        return model_class

    @classmethod
    def get_model(cls, model_name: str) -> Type:
        return cls.register(model_name)
