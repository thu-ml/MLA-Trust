import json
import os

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import BaseChat


class PixtralChat(BaseChat):
    def __init__(self, model_name="mistral-community/pixtral-12b"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, device_map="cuda"
        )

        chat_template_path = "models/chat_template/pixtral.json"
        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r") as fp:
                self.chat_template = json.load(fp)["chat_template"]
        else:
            self.chat_template = None

    def prepare_inputs(self, chat):
        convs = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            image_urls = [image_dict["image_url"]["url"] for image_dict in content[1:]]
            parsed_content = [{"type": "text", "content": prompt}] + [
                {"type": "image", "url": image_url} for image_url in image_urls
            ]
            convs.append({"role": role, "content": parsed_content})

        inputs = self.processor.apply_chat_template(
            convs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            chat_template=self.chat_template,
        ).to(self.model.device)
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
            generate_ids = self.model.generate(**inputs, **generation_args)

        generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response
