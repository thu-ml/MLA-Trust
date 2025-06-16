import copy
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from MobileAgentE.agents import ExperienceRetrieverShortCut
from MobileAgentE.api import encode_image

from .base import BaseChat

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))


class OpenAIChat(BaseChat):
    def __init__(self, model_name="gpt-4o-2024-11-20", retry=1):
        self.model_name = model_name
        if "claude" in model_name:
            self.base_url = os.getenv("CLAUDE_API_URL")
            self.api_key = os.getenv("CLAUDE_API_KEY")
        else:
            self.base_url = os.getenv("OPENAI_API_URL")
            self.api_key = os.getenv("OPENAI_API_KEY")

        print(f"Using {model_name} model")
        print(f"Using {self.base_url} as API URL")
        print(f"Using {self.api_key} as API Key")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.retry = retry

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
            "timeout": 3600,
        }
        generation_args.update(custom_generation_args)

        retry = self.retry
        while retry > 0:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=msgs, **generation_args
                )
                response = response.choices[0].message.content
                break
            except Exception as e:
                print(e)
                import traceback

                traceback.print_exception(e)
                response = "An error occurred during generation:\n{}".format(e)
                retry -= 1

        return response


if __name__ == "__main__":
    chatmodel = OpenAIChat(model_name="gpt-4-turbo")
    # chatmodel = OpenAIChat(model_name="gemini-2.0-flash")
    # chatmodel = OpenAIChat(model_name="gemini-2.0-pro-exp-02-05")
    # chatmodel = OpenAIChat(model_name="claude-3-7-sonnet-20250219")
    agent_pipeline = False

    if agent_pipeline:
        instruction = "Find me at least two survery papers on Large Language Models. Check the detailed abstract of the most cited one. And then create a new note in Notes and add the titles the papers you found. Also include the abstract of the most cited paper."
        initial_shortcuts = {
            "Tap_Type_and_Enter": {
                "name": "Tap_Type_and_Enter",
                "arguments": ["x", "y", "text"],
                "description": 'Tap an input box at position (x, y), Type the "text", and then perform the Enter operation. Very useful for searching and sending messages!',
                "precondition": "There is a text input box on the screen with no previously entered content.",
                "atomic_action_sequence": [
                    {"name": "Tap", "arguments_map": {"x": "x", "y": "y"}},
                    {"name": "Type", "arguments_map": {"text": "text"}},
                    {"name": "Enter", "arguments_map": {}},
                ],
            }
        }

        experience_retriever_shortcut = ExperienceRetrieverShortCut()
        experience_retriever_shortcut_prompt = experience_retriever_shortcut.get_prompt(
            instruction, initial_shortcuts
        )
        chat_experience_retrieval_shortcut = experience_retriever_shortcut.init_chat()
        chat_experience_retrieval_shortcut = chatmodel.add_response(
            "user",
            experience_retriever_shortcut_prompt,
            chat_experience_retrieval_shortcut,
            images=None,
        )
    else:
        experience_retriever_shortcut = ExperienceRetrieverShortCut()
        chat_convs = experience_retriever_shortcut.init_chat(
            sysetm_prompt="You are a helpful assistant."
        )
        # instruction = "Who are you"
        # chat_convs = chatmodel.add_response(
        #     "user",
        #     instruction,
        #     chat_convs,
        #     images=None,
        # )

        instruction = "Tell me a joke."
        chat_convs = chatmodel.add_response(
            "user",
            instruction,
            chat_convs,
            images=None,
        )

        # instruction = "Descibe this image."
        # chat_convs = chatmodel.add_response(
        #     "user",
        #     instruction,
        #     chat_convs,
        #     images="dev_imgs/image_01.jpg",
        # )

        # instruction = "Descibe difference between these two images."
        # chat_convs = chatmodel.add_response(
        #     "user",
        #     instruction,
        #     chat_convs,
        #     images=[
        #         "dev_imgs/image_01.jpg",
        #         "dev_imgs/image_02.jpg",
        #     ],
        # )

    response = chatmodel.generate(chat_convs)
    print(response)
