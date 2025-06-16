import base64
import copy
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from .base import BaseChat

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIChat(BaseChat):
    def __init__(self, model_name="gpt-4o-2024-11-20"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def add_response(self, role, prompt, chat_history, images=None):
        new_chat_history = copy.deepcopy(chat_history)

        if images is not None:
            if not isinstance(images, List):
                images = [images]
        else:
            images = []

        content = [
            {"type": "text", "text": prompt},
        ] + [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"},
            }
            for image in images
        ]

        new_chat_history.append([role, content])
        return new_chat_history

    def prepare_inputs(self, chat):
        convs = []
        for role, content in chat:
            convs.append({"role": role, "content": content})
        return convs

    def generate(self, chat, custom_generation_args={}):
        msgs = self.prepare_inputs(chat)
        generation_args = {
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        generation_args.update(custom_generation_args)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=msgs, **generation_args
            )
            response = response.choices[0].message.content
        except Exception as e:
            import traceback

            print(traceback.print_exception(e))
            from pprint import pp

            pp(msgs)
            response = None

        return response
