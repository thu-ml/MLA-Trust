import os

from dotenv import load_dotenv

from src.scene.web.src.demo_utils.inference_unified_engine import UnifiedEngine

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    model = args.model

    temperature = args.temperature
    engine = UnifiedEngine(model=model, temperature=temperature)

    system_prompt = "You are a helpful assistant."
    user_prompt = "Describe this image."
    prompt = [system_prompt, user_prompt]
    output = engine.generate(prompt=prompt, image_path="snowman.jpg", turn_number=0)
    print(output)
