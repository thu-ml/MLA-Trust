import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseChat


class MiniCPMOChat(BaseChat):
    def __init__(self, model_name="openbmb/MiniCPM-o-2_6"):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )  # sdpa or flash_attention_2, no eager
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model.init_tts()
        self.model.tts.float()

    def prepare_inputs(self, chat):
        convs = []
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            pil_images = [
                Image.open(image_dict["image_url"]["url"]) for image_dict in content[1:]
            ]
            parsed_content = pil_images + [prompt]
            convs.append({"role": role, "content": parsed_content})

        return convs

    def generate(self, chat, custom_generation_args={}):
        msgs = self.prepare_inputs(chat)

        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        generation_args.update(custom_generation_args)

        with torch.inference_mode():
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                **generation_args,
            )

        return response
