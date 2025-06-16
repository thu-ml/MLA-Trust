import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from MobileAgentE.agents import ExperienceRetrieverShortCut

from .base import BaseChat


class QwenChat(BaseChat):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def prepare_inputs(self, chat):
        convs = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            image_urls = [image_dict["image_url"]["url"] for image_dict in content[1:]]
            parsed_content = [
                {"type": "image", "image": image_url} for image_url in image_urls
            ] + [{"type": "text", "text": prompt}]
            convs.append({"role": role, "content": parsed_content})

        text = self.processor.apply_chat_template(
            convs,
            add_generation_prompt=True,
            tokenize=False,
        )
        image_inputs, video_inputs = process_vision_info(convs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    def generate(self, chat, custom_generation_args={}):
        inputs = self.prepare_inputs(chat)
        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        generation_args.update(custom_generation_args)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_args)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response


if __name__ == "__main__":
    chatmodel = QwenChat()
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
        #     images="dev_imgs/snowman.jpg",
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
