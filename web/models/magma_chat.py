import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import BaseChat


class MagmaChat(BaseChat):
    def __init__(self, model_name="microsoft/Magma-8B"):
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.to("cuda")

    def prepare_inputs(self, chat):
        convs = []
        pil_images = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            query = content[0]["text"]
            pil_images_round = [
                Image.open(image_dict["image_url"]["url"]) for image_dict in content[1:]
            ]
            pil_images.extend(pil_images_round)

            parsed_content = ""
            for _ in range(len(pil_images_round)):
                parsed_content += "<image_start><image><image_end>\n\n"
            parsed_content += query
            convs.append({"role": role, "content": parsed_content})

        prompt = self.processor.tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )

        if len(pil_images) == 0:
            pil_images = None
        inputs = self.processor(images=pil_images, texts=prompt, return_tensors="pt")

        if "pixel_values" in inputs.keys():
            inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
        if "image_sizes" in inputs.keys():
            inputs["image_sizes"] = inputs["image_sizes"].unsqueeze(0)
        inputs = inputs.to("cuda").to(self.dtype)

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
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        return response
