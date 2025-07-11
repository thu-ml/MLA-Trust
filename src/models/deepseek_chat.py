import torch
from transformers import AutoModelForCausalLM

from src.models.src.deepseek_vl2.deepseek_vl2.models import (
    DeepseekVLV2ForCausalLM,
    DeepseekVLV2Processor,
)
from src.models.src.deepseek_vl2.deepseek_vl2.utils.io import load_pil_images

from .base import BaseChat


class DeepSeekVL2Chat(BaseChat):
    def __init__(self, model_name="deepseek-ai/deepseek-vl2"):
        self.vl_chat_processor: DeepseekVLV2Processor = (
            DeepseekVLV2Processor.from_pretrained(model_name)
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
        )
        self.vl_gpt = vl_gpt.to(torch.bfloat16).eval()

    def prepare_inputs(self, chat):
        """
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\nDescribe this image.",
                "images": ["dev_imgs/snowman.jpg"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        """
        role_map = {
            "user": "<|User|>",
            "assistant": "<|Assistant|>",
        }
        convs = []
        system_prompt = ""
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            if role == "system":
                assert system_prompt == "", (
                    f"system prompt should be empty, but got {system_prompt}"
                )
                system_prompt = content[0]["text"]
                continue

            prompt = content[0]["text"]
            content_str = ""
            for data_dict in content[1:]:
                content_str += "<image>\n"
            content_str += prompt
            images = [data_dict["image_url"]["url"] for data_dict in content[1:]]

            conv = {"role": role_map[role], "content": content_str}
            if images:
                conv["images"] = images
            convs.append(conv)
        convs.append({"role": "<|Assistant|>", "content": ""})
        pil_images = load_pil_images(convs)

        prepare_inputs = self.vl_chat_processor(
            conversations=convs,
            images=pil_images,
            # force_batchify=True,
            system_prompt=system_prompt,
        ).to(self.vl_gpt.device)

        return prepare_inputs

    def generate(self, chat, custom_generation_args={}):
        inputs = self.prepare_inputs(chat)
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**inputs)
        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        generation_args.update(custom_generation_args)

        with torch.inference_mode():
            outputs = self.vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_args,
            )
            response = self.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )

        return response
