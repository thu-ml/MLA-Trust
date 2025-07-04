import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import BaseChat

__all__ = ["InternVL2Chat"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL2Chat(BaseChat):
    def __init__(self, model_name="OpenGVLab/InternVL2-8B"):
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )

    def prepare_inputs(self, chat):
        if len(chat) >= 1 and chat[0][0] == "system":
            system_content = chat[0][1]
            assert len(system_content) == 1 and system_content[0]["type"] == "text"

            self.model.system_message = system_content[0]["text"]
            chat = chat[1:]

        pixel_values_list = []
        history = []

        assert len(chat) % 2 == 1
        for turn_id, (role, content) in enumerate(chat):
            # sanity check
            assert content[0]["type"] == "text"
            for data_dict in content[1:]:
                assert data_dict["type"] == "image_url", f"{data_dict}"

            prompt = content[0]["text"]
            image_urls = [image_dict["image_url"]["url"] for image_dict in content[1:]]
            if len(image_urls) > 1:
                image_placeholders = ""
                for i in range(len(image_urls)):
                    image_placeholders += f"Image-{i + 1}: <image>\n"
            elif len(image_urls) == 1:
                image_placeholders = "<image>\n"
            else:
                image_placeholders = ""

            prompt = image_placeholders + prompt
            """
            history = [
                (user_prompt, assistant_response),
                (user_prompt, assistant_response),
                (user_prompt, assistant_response),
                ...
                (user_prompt, ),
            ]
            """
            if turn_id % 2 == 0:
                assert role == "user"
                history.append((prompt,))
            else:
                assert role == "assistant"
                history[-1] = (history[-1], prompt)

            for image_url in image_urls:
                pixel_values_list.append(
                    load_image(image_url, max_num=12).to(torch.bfloat16).cuda()
                )

        if len(pixel_values_list) > 1:
            num_patches_list = [
                pixel_value.size(0) for pixel_value in pixel_values_list
            ]
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif len(pixel_values_list) == 1:
            num_patches_list = None
            pixel_values = pixel_values_list[0]
        else:
            num_patches_list = None
            pixel_values = None

        # get user prompt
        question = history[-1][0]
        history = history[:-1]
        if not history:
            history = None

        return history, question, pixel_values, num_patches_list

    def generate(self, chat, custom_generation_args={}):
        history, question, pixel_values, num_patches_list = self.prepare_inputs(chat)
        generation_args = {
            "max_new_tokens": 512,
            "do_sample": False,
        }

        generation_args.update(custom_generation_args)

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            history=history,
            generation_config=generation_args,
            num_patches_list=num_patches_list,
            return_history=True,
        )

        return response
