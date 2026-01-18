from abc import ABC, abstractmethod

import openai
import os

from pydantic import BaseModel

from llm_rpg.llm.llm_cost_tracker import LLMCostTracker
from ollama import chat
from ollama import ChatResponse
from llm_rpg.utils.logger import get_logger

logger = get_logger(__name__)


class LLM(ABC):
    @abstractmethod
    def generate_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_structured_completion(
        self, prompt: str, output_schema: BaseModel
    ) -> BaseModel:
        pass


class GroqLLM(LLM):
    def __init__(
        self,
        llm_cost_tracker: LLMCostTracker,
        model: str = "llama-3.3-70b-versatile",
    ):
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set")
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.model = model
        self.pricing = {
            "llama-3.3-70b-versatile": {
                "input_token_price": 0.59 / 1000000,
                "output_token_price": 0.79 / 1000000,
            },
            "openai/gpt-oss-20b": {
                "input_token_price": 0.075 / 1000000,
                "output_token_price": 0.30 / 1000000,
            },
            "openai/gpt-oss-120b": {
                "input_token_price": 0.15 / 1000000,
                "output_token_price": 0.60 / 1000000,
            },
        }
        self.llm_cost_tracker = llm_cost_tracker

    def generate_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        self._calculate_completion_costs(response)
        return response.choices[0].message.content

    def _calculate_completion_costs(self, response: openai.types.Completion):
        input_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        input_cost = input_tokens * self.pricing[self.model]["input_token_price"]
        completion_cost = (
            completion_tokens * self.pricing[self.model]["output_token_price"]
        )

        self.llm_cost_tracker.add_cost(
            input_tokens, completion_tokens, input_cost, completion_cost
        )

    def generate_structured_completion(
        self, prompt: str, output_schema: BaseModel
    ) -> BaseModel:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        parsed_output = output_schema.model_validate_json(
            response.choices[0].message.content
        )
        self._calculate_completion_costs(response)
        return parsed_output


class OllamaLLM(LLM):
    def __init__(
        self,
        llm_cost_tracker: LLMCostTracker,
        model: str,
    ):
        self.model = model
        self.llm_cost_tracker = llm_cost_tracker

    def _calculate_completion_costs(self, response: ChatResponse):
        input_tokens = response.prompt_eval_count
        completion_tokens = response.eval_count

        self.llm_cost_tracker.add_cost(input_tokens, completion_tokens, 0, 0)

    def generate_completion(self, prompt: str) -> str:
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            think=False,
        )
        self._calculate_completion_costs(response)
        return response.message.content

    def generate_structured_completion(
        self, prompt: str, output_schema: BaseModel
    ) -> BaseModel:
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=output_schema.model_json_schema(),
            think=False,
        )
        parsed_output = output_schema.model_validate_json(response.message.content)
        self._calculate_completion_costs(response)
        return parsed_output


class ZaiLLM(LLM):
    def __init__(
        self,
        llm_cost_tracker: LLMCostTracker,
        model: str = "glm-4.7",
    ):
        if not os.environ.get("ZAI_API_KEY"):
            raise ValueError("ZAI_API_KEY is not set")
        self.client = openai.OpenAI(
            base_url="https://api.z.ai/api/coding/paas/v4/",
            api_key=os.environ.get("ZAI_API_KEY"),
            timeout=180.0,
        )
        self.model = model
        self.pricing = {
            "glm-4.7": {
                "input_token_price": 0.60 / 1000000,
                "output_token_price": 2.20 / 1000000,
            },
            "glm-4.6": {
                "input_token_price": 0.60 / 1000000,
                "output_token_price": 2.20 / 1000000,
            },
            "glm-4.5": {
                "input_token_price": 0.60 / 1000000,
                "output_token_price": 2.20 / 1000000,
            },
            "glm-4.5-flash": {
                "input_token_price": 0.0,
                "output_token_price": 0.0,
            },
            "glm-4.5-air": {
                "input_token_price": 0.20 / 1000000,
                "output_token_price": 1.10 / 1000000,
            },
        }
        self.llm_cost_tracker = llm_cost_tracker

    def generate_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        self._calculate_completion_costs(response)
        return response.choices[0].message.content

    def _calculate_completion_costs(self, response: openai.types.Completion):
        input_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        model_pricing = self.pricing.get(
            self.model, {"input_token_price": 0.0, "output_token_price": 0.0}
        )
        input_cost = input_tokens * model_pricing["input_token_price"]
        completion_cost = completion_tokens * model_pricing["output_token_price"]

        self.llm_cost_tracker.add_cost(
            input_tokens, completion_tokens, input_cost, completion_cost
        )

    def generate_structured_completion(
        self, prompt: str, output_schema: BaseModel
    ) -> BaseModel:
        try:
            logger.debug(
                f"Calling Zai API model={self.model}, prompt length={len(prompt)}"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            logger.debug(f"ZaiLLM raw response:\n{content[:500]}...")
            parsed_output = output_schema.model_validate_json(content)
            self._calculate_completion_costs(response)
            logger.debug(f"ZaiLLM structured completion successful")
            return parsed_output
        except Exception as e:
            logger.error(
                f"ZaiLLM structured completion error: {type(e).__name__}: {e}",
                exc_info=True,
            )
            if hasattr(e, "response"):
                logger.error(f"ZaiLLM API response: {e.response}")
            raise
