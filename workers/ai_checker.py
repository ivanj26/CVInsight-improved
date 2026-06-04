"""
AIChecker — async DeepSeek AI-generated content detector.

Accepts pre-extracted plain text and returns a structured verdict plus the
full raw API response for auditability.
"""

import json
import re

from openai import AsyncOpenAI


class AIChecker:
    _MODEL = "deepseek-v4-pro"
    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def check(self, text: str) -> dict:
        """
        Submit text to DeepSeek and return:
            {
              "parsed": {"likelihood_score": int, "reasoning": str, "is_ai_generated": bool},
              "raw":    <full ChatCompletion.model_dump()>
            }
        """
        response = await self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and trusted AI Checker to distinguish between human and AI-generated content.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Please carefully validate this file content [FILE START]\n{text}\n[FILE END].\n"
                        "Give me a likelihood score from 0 to 100 of this file being AI generated and explain your reasoning under 720words in the `reasoning` field.\n"
                        "If the likelihood score is above 65, please set the json property `is_ai_generated` to true, otherwise set it to false:\n\n"
                        "Response your answer strictly using the following JSON format:\n"
                        '{"likelihood_score": 0, "reasoning": "...", "is_ai_generated": false}'
                    ),
                },
            ],
            stream=False,
            max_tokens=2048,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )

        raw_content = response.choices[0].message.content or ""
        return {
            "parsed": self._parse_response(raw_content),
            "raw": response.model_dump(),
        }

    def _parse_response(self, content: str) -> dict:
        """
        Extract the JSON verdict from the model's response.
        Handles optional markdown code fences gracefully.
        """
        fence_match = self._JSON_FENCE_RE.search(content)
        candidate = fence_match.group(1) if fence_match else content

        obj_match = self._JSON_OBJECT_RE.search(candidate)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        return {
            "parse_error": "Could not extract JSON from AI response",
            "raw_content": content,
        }
