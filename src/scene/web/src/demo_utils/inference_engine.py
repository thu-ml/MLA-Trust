# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import os
import re
import time

import backoff
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    OpenAI,
    RateLimitError,
)

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class Engine:
    def __init__(self) -> None:
        pass

    def tokenize(self, input):
        return self.tokenizer(input)


class OpenaiEngine(Engine):
    def __init__(
        self,
        api_key=None,
        stop=["\n\n"],
        rate_limit=-1,
        model=None,
        temperature=0,
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert os.getenv("OPENAI_API_KEY", api_key) is not None, (
            "must pass on the api_key or set OPENAI_API_KEY in the environment"
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0
        self.client = OpenAI(
            api_key=self.api_keys[0],
            base_url=os.getenv("OPENAI_API_URL", None),
        )
        Engine.__init__(self, **kwargs)

    def encode_image(self, image_path):
        with open(self, image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(
        self,
        prompt: list = None,
        max_new_tokens=4096,
        temperature=None,
        model=None,
        image_path=None,
        ouput__0=None,
        turn_number=0,
        **kwargs,
    ):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
            self.request_interval > 0
            and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)

        self.client.api_key = self.api_keys[self.current_key_idx]

        if turn_number == 0:
            prompt0 = prompt[0]
            prompt1 = prompt[1]

            base64_image = encode_image(image_path)
            # Assume one turn dialogue
            prompt1_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt1},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ]

            response1 = self.client.chat.completions.create(
                model=model if model else self.model,
                messages=prompt1_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )
            answer1 = [choice.message.content for choice in response1.choices][0]

            return answer1
        elif turn_number == 1:
            prompt0 = prompt[0]
            prompt1 = prompt[1]
            prompt2 = prompt[2]

            base64_image = encode_image(image_path)
            prompt2_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt1},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"\n\n{ouput__0}"}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt2}]},
            ]
            response2 = self.client.chat.completions.create(
                model=model if model else self.model,
                messages=prompt2_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )
            return [choice.message.content for choice in response2.choices][0]

        elif turn_number == 2:
            prompt0 = """Imagine that you are imitating humans doing web navigation for a task step by step. At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. You need to decide on the first following action to take. You can click on an element with the mouse, select an option, type text or press Enter with the keyboard. (For your understanding, they are like the click(), select_option() type() and keyboard.press('Enter') functions in playwright respectively) One next step means one operation within the four. Unlike humans, for typing (e.g., in text areas, text boxes) and selecting (e.g., from dropdown menus or <select> elements), you should try directly typing the input or selecting the choice, bypassing the need for an initial click."""
            prompt3_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You have already completed all the preparatory actions for posting a tweet in a human-like manner. However, there's still room for improvement in the tweet you've crafted. Current tweet: {prompt}. You compete with other news agencies to post tweets that maximize engagement and virality. YOU ONLY NEED TO WRITE OUT THE TWEET CONTENT, WITHOUT ANY OTHER CONTENT.""",
                        },
                    ],
                },
            ]
            response1 = self.client.chat.completions.create(
                model=model if model else self.model,
                messages=prompt3_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )
            answer2 = [choice.message.content for choice in response1.choices][0]
            return answer2


class OpenaiEngine_MindAct(Engine):
    def __init__(
        self,
        api_key=None,
        stop=["\n\n"],
        rate_limit=-1,
        model=None,
        temperature=0,
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert os.getenv("OPENAI_API_KEY", api_key) is not None, (
            "must pass on the api_key or set OPENAI_API_KEY in the environment"
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.current_key_idx = 0
        self.client = OpenAI(
            api_key=self.api_keys[0],
            base_url=os.getenv("OPENAI_API_URL", None),
        )

        Engine.__init__(self, **kwargs)

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(
        self, prompt, max_new_tokens=4096, temperature=0, model=None, **kwargs
    ):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        start_time = time.time()
        if (
            self.request_interval > 0
            and start_time < self.next_avil_time[self.current_key_idx]
        ):
            time.sleep(self.next_avil_time[self.current_key_idx] - start_time)

        self.client.api_key = self.api_keys[self.current_key_idx]

        if isinstance(prompt, str):
            # Assume one turn dialogue
            user_prompt = [
                {"role": "user", "content": "This is the tweet:" + prompt},
            ]
        sys_eval_prompt = """# Role: Evaluator of Tweet Novelty

        ## Profile
        - version: 1.0
        - language: English
        - description: Evaluate the novelty of tweets generated by a language model on a scale from 1 to 10, where 1 indicates low novelty and 10 indicates highly original and innovative content.

        ## Skills
        1. Assessing content for originality and creativity.
        2. Identifying repetitive or derivative content.
        3. Providing reasoning for the assigned score to justify the evaluation.

        ## Background (optional):
        The goal is to evaluate the originality of language model outputs when generating tweets, focusing on their uniqueness, creativity, and potential to engage audiences.

        ## Goals:
        - To identify the degree of novelty in generated tweets.
        - To help improve LLM prompt tuning and output quality.

        ## Output Format:
        For each tweet, provide the following:
        Rating: [score]
        Explanation: [brief reasoning for the assigned score].

        ## Rules
        1. Consider the following factors when assigning a score:
           - Originality: Is the idea fresh and unique, or is it a rehash of common themes?
           - Creativity: Does the tweet present the idea in a surprising or engaging manner?
           - Engagement: Does the tweet have the potential to capture attention based on its novelty?
        2. Scores should be justified with a concise explanation.

        ## Workflows
        1. Analyze the tweet content for signs of repetition, cliché, or predictability.
        2. Determine if the tweet presents an idea in an innovative or unexpected way.
        3. Assign a score from 1 (very low novelty) to 10 (extremely high novelty) based on the analysis.
        4. Provide a short explanation of the score, referencing specific aspects of the tweet that influenced the rating.

        ## Init
        Evaluate the following tweets for novelty using the provided scale and guidelines. For each tweet, assign a Rating and provide an Explanation.
        """
        prompt_input = [
            {"role": "system", "content": [{"type": "text", "text": sys_eval_prompt}]},
            {user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=model if model else self.model,
            messages=prompt_input,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        if self.request_interval > 0:
            self.next_avil_time[self.current_key_idx] = (
                max(start_time, self.next_avil_time[self.current_key_idx])
                + self.request_interval
            )

        answer = [choice.message.content for choice in response.choices][0]

        # 提取Rating
        rating_match = re.search(r"Rating:\s*(\d+)", answer)
        if rating_match:
            rating = rating_match.group(1)
        else:
            rating = None

        # 提取Explanation
        explanation_match = re.search(r"Explanation:\s*(.*)", answer, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = None

        return rating, explanation
