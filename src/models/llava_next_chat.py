import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import BaseChat


class LlavaNextChat(BaseChat):
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def prepare_inputs(self, chat):
        convs = []
        pil_images = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            pil_images_round = [
                Image.open(image_dict["image_url"]["url"]) for image_dict in content[1:]
            ]
            pil_images.extend(pil_images_round)
            parsed_content = [
                {"type": "image"} for _ in range(len(pil_images_round))
            ] + [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
            convs.append({"role": role, "content": parsed_content})

        return convs, pil_images

    def generate(self, chat, custom_generation_args={}):
        msgs, pil_images = self.prepare_inputs(chat)

        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        generation_args.update(custom_generation_args)

        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        prompts = [prompt]

        # We can simply feed images in the order they have to be used in the text prompt
        # Each "<image>" token uses one image leaving the next for the subsequent "<image>" tokens
        inputs = self.processor(
            images=pil_images if len(pil_images) > 0 else None,
            text=prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, max_new_tokens=512)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return response
