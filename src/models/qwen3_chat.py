import torch
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration

from .base import BaseChat


class Qwen3VLChat(BaseChat):
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        # follow ModelScope official usage; device placement handled by device_map
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )

    def prepare_inputs(self, chat):
        messages = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            image_urls = [image_dict["image_url"]["url"] for image_dict in content[1:]]

            parsed_content = (
                [{"type": "image", "image": image_url} for image_url in image_urls]
                + [{"type": "text", "text": prompt}]
                if image_urls
                else [{"type": "text", "text": prompt}]
            )
            messages.append({"role": role, "content": parsed_content})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Ensure inputs are on the same device as the model to avoid device mismatch warnings.
        device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(device)

        return inputs

    def generate(self, chat, custom_generation_args={}):
        inputs = self.prepare_inputs(chat)
        generation_args = {
            # align with playground/qwen3.py default
            "max_new_tokens": 128,
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
