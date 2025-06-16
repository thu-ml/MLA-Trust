from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from MobileAgentE.agents import ExperienceRetrieverShortCut

from .base import BaseChat


class PhiChat(BaseChat):
    def __init__(self, model_name="microsoft/Phi-4-multimodal-instruct"):
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation="flash_attention_2",
        ).cuda()

        # Load generation config
        self.generation_config = GenerationConfig.from_pretrained(model_name)

    def prepare_inputs(self, chat):
        messages = []
        images = []
        image_cnt = 0
        for role, content in chat:
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            image_urls = [image_dict["image_url"]["url"] for image_dict in content[1:]]

            placeholder = ""
            for image_url in image_urls:
                image_cnt += 1
                placeholder += f"<|image_{image_cnt}|>\n"
                images.append(Image.open(image_url))

            messages.append({"role": role, "content": placeholder + prompt})

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, images, return_tensors="pt").to(
            self.model.device
        )

        return inputs

    def generate(self, chat, custom_generation_args={}):
        inputs = self.prepare_inputs(chat)
        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        generation_args.update(custom_generation_args)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            generation_config=self.generation_config,
            num_logits_to_keep=1,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response


if __name__ == "__main__":
    chatmodel = PhiChat()
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

        # instruction = "Tell me a joke."
        # chat_convs = chatmodel.add_response(
        #     "user",
        #     instruction,
        #     chat_convs,
        #     images=None,
        # )

        # instruction = "Descibe this image."
        # chat_convs = chatmodel.add_response(
        #     "user",
        #     instruction,
        #     chat_convs,
        #     images="dev_imgs/snowman.jpg",
        # )

        instruction = "Descibe difference between these two images."
        chat_convs = chatmodel.add_response(
            "user",
            instruction,
            chat_convs,
            images=[
                "dev_imgs/image_01.jpg",
                "dev_imgs/image_02.jpg",
            ],
        )

    response = chatmodel.generate(chat_convs)
    print(response)
