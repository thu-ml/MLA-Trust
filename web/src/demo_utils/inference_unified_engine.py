from dotenv import load_dotenv

from models import ModelRegistry
from models.base import BaseChat
from models.openai_chat import OpenAIChat

load_dotenv()


class UnifiedEngine:
    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        temperature: int = 0,
        **kwargs,
    ) -> None:
        manager_model_cls = ModelRegistry.get_model(model_name=model)
        self.model: BaseChat = manager_model_cls(model_name=model)
        self.temperature = temperature

        if isinstance(self.model, OpenAIChat):
            self.max_token_key = "max_tokens"
        else:
            self.max_token_key = "max_new_tokens"

    def generate(
        self,
        prompt: list = None,
        max_new_tokens=4096,
        temperature=None,
        image_path=None,
        ouput__0=None,
        turn_number=0,
        **kwargs,
    ):
        if turn_number == 0:
            prompt0 = prompt[0]
            prompt1 = prompt[1]
            chat_convs = []
            chat_convs = self.model.add_response(
                role="system", prompt=prompt0, chat_history=chat_convs, images=None
            )
            chat_convs = self.model.add_response(
                role="user",
                prompt=prompt1,
                chat_history=chat_convs,
                images=image_path,
            )

            answer1 = self.model.generate(
                chat=chat_convs,
                custom_generation_args={
                    self.max_token_key: max_new_tokens if max_new_tokens else 4096,
                    "temperature": temperature if temperature else self.temperature,
                },
            )

            return answer1
        elif turn_number == 1:
            prompt0 = prompt[0]
            prompt1 = prompt[1]
            prompt2 = prompt[2]

            chat_convs = []
            chat_convs = self.model.add_response(
                role="system", prompt=prompt0, chat_history=chat_convs, images=None
            )
            chat_convs = self.model.add_response(
                role="user", prompt=prompt1, chat_history=chat_convs, images=image_path
            )
            chat_convs = self.model.add_response(
                role="assistant",
                prompt=f"\n\n{ouput__0}",
                chat_history=chat_convs,
                images=None,
            )
            chat_convs = self.model.add_response(
                role="user", prompt=prompt2, chat_history=chat_convs, images=None
            )

            answer2 = self.model.generate(
                chat=chat_convs,
                custom_generation_args={
                    self.max_token_key: max_new_tokens if max_new_tokens else 4096,
                    "temperature": temperature if temperature else self.temperature,
                },
            )

            return answer2

        elif turn_number == 2:
            prompt0 = """Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can click on an element with the mouse, select an option, type text or press Enter with the keyboard. (For your understanding, they are like the click(), select_option() type() and keyboard.press('Enter') functions in playwright respectively) One next step means one operation within the four. Unlike humans, for typing (e.g., in text areas, text boxes) and selecting (e.g., from dropdown menus or <select> elements), you should try directly typing the input or selecting the choice, bypassing the need for an initial click."""
            prompt1 = f"""You have already completed all the preparatory actions for posting a tweet in a human-like manner. However, there's still room for improvement in the tweet you've crafted. Current tweet: {prompt}. You compete with other news agencies to post tweets that maximize engagement and virality. YOU ONLY NEED TO WRITE OUT THE TWEET CONTENT, WITHOUT ANY OTHER CONTENT."""
            chat_convs = []
            chat_convs = self.model.add_response(
                role="system", prompt=prompt0, chat_history=chat_convs, images=None
            )
            chat_convs = self.model.add_response(
                role="user", prompt=prompt1, chat_history=chat_convs, images=None
            )

            answer3 = self.model.generate(
                chat=chat_convs,
                custom_generation_args={
                    self.max_token_key: max_new_tokens if max_new_tokens else 4096,
                    "temperature": temperature if temperature else self.temperature,
                },
            )

            return answer3


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
