import gc

import torch

from src.models import ModelRegistry
from src.scene.mobile.MobileAgentE.agents import ExperienceRetrieverShortCut


def test_models(model_name):
    model_cls = ModelRegistry.get_model(model_name)
    model = model_cls(model_name=model_name)

    experience_retriever_shortcut = ExperienceRetrieverShortCut()
    chat_convs = experience_retriever_shortcut.init_chat(
        sysetm_prompt="You are a helpful assistant."
    )

    # instruction = "Tell me a joke."
    # chat_convs = model.add_response(
    #     "user",
    #     instruction,
    #     chat_convs,
    #     images=None,
    # )

    instruction = "Descibe this image."
    chat_convs = model.add_response(
        "user",
        instruction,
        chat_convs,
        images="playground/data/snowman.jpg",
    )

    # instruction = "Descibe difference between these two images."
    # chat_convs = model.add_response(
    #     "user",
    #     instruction,
    #     chat_convs,
    #     images=[
    #         "playground/data/image1.png",
    #         "playground/data/image2.png",
    #     ],
    # )

    response = model.generate(chat_convs)
    print(response)

    print("success get model: ", model_name)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20")
    args = parser.parse_args()
    model_name = args.model

    test_models(model_name)
