import copy
from abc import ABC, abstractmethod
from typing import List


class BaseChat(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def add_response(self, role, prompt, chat_history, images=None):
        new_chat_history = copy.deepcopy(chat_history)

        if images is not None:
            if not isinstance(images, List):
                images = [images]
        else:
            images = []

        content = [
            {"type": "text", "text": prompt},
        ] + [{"type": "image_url", "image_url": {"url": image}} for image in images]

        new_chat_history.append([role, content])
        return new_chat_history

    @abstractmethod
    def generate(self, chat, custom_generation_args={}):
        pass
