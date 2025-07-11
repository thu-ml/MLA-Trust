import copy
import warnings

import torch
from PIL import Image

from src.models.src.llava_next.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from src.models.src.llava_next.llava.conversation import conv_templates
from src.models.src.llava_next.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from src.models.src.llava_next.llava.model.builder import load_pretrained_model

from .base import BaseChat

warnings.filterwarnings("ignore")


class LlavaOVChat(BaseChat):
    def __init__(self, model_name="lmms-lab/llava-onevision-qwen2-72b-ov-sft"):
        self.device = "cuda"
        self.device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                model_name, None, "llava_qwen", device_map=self.device_map
            )
        )  # Add any other thing you want to pass in llava_model_args

        self.model.eval()

    def prepare_inputs(self, chat):
        conv_template = (
            "qwen_1_5"  # Make sure you use correct chat template for different models
        )
        conv = copy.deepcopy(conv_templates[conv_template])
        images = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            if role == "system":
                conv.system = f"""<|im_start|>system\n{content[0]["text"]}"""
                continue

            prompt = content[0]["text"]
            if len(content) > 1:
                question = ""
                for _ in range(len(content[1:])):
                    question += f"{DEFAULT_IMAGE_TOKEN}\n"
                question += prompt
            else:
                question = prompt

            conv.append_message(conv.roles[0], question)

            # Process images
            for image_dict in content[1:]:
                image = Image.open(image_dict["image_url"]["url"])
                images.append(image)

        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        return prompt_question, images

    def generate(self, chat, custom_generation_args={}):
        prompt_question, images = self.prepare_inputs(chat)

        generation_args = {
            "max_new_tokens": 4096,
            "do_sample": False,
            "use_cache": True,
        }

        generation_args.update(custom_generation_args)

        input_ids = (
            tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [
            _image.to(dtype=torch.float16, device=self.device)
            for _image in image_tensors
        ]
        image_sizes = [image.size for image in images]

        with torch.inference_mode():
            cont = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                **generation_args,
            )
            response = self.tokenizer.batch_decode(
                cont,
                skip_special_tokens=True,
            )[0]

        return response
