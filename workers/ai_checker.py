"""
AIChecker — async TokenRouter AI-generated content detector.

Accepts pre-extracted plain text and returns a structured verdict plus the
full raw API response for auditability.
"""

import json
import os
import re

from openai import AsyncOpenAI


_VERDICT_SCHEMA = {
    "name": "ai_check_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "likelihood_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "reasoning": {"type": "string"},
            "is_ai_generated": {"type": "boolean"},
        },
        "required": ["likelihood_score", "reasoning", "is_ai_generated"],
        "additionalProperties": False,
    },
}


class AIChecker:
    _MODEL = os.environ.get("DOCS_CHECKER_DEFAULT_LLM_MODEL", "agentrouter/glm-5.3")
    _FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(self, api_key: str, base_url: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def check(self, text: str) -> dict:
        """
        Submit text to TokenRouter and return:
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
                        "Reply with ONLY a ```json fenced code block containing exactly this JSON format:\n"
                        '{"likelihood_score": 0, "reasoning": "...", "is_ai_generated": false}'
                    ),
                },
            ],
            stream=False,
            max_completion_tokens=4096,
            reasoning_effort="high",
            response_format={"type": "json_schema", "json_schema": _VERDICT_SCHEMA},
            extra_body={"thinking": {"type": "enabled"}},
        )

        choice = response.choices[0]
        raw_content = choice.message.content or ""
        parsed = self._parse_response(raw_content)
        if choice.finish_reason == "length":
            parsed.setdefault("truncated", True)

        return {
            "parsed": parsed,
            "raw": response.model_dump(),
        }

    def _parse_response(self, content: str) -> dict:
        """
        Extract the JSON verdict from the model's response.
        Handles markdown code fences and a leading token eaten by the gateway.
        """
        candidate = self._strip_fences(content)

        obj_match = self._JSON_OBJECT_RE.search(candidate)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        repaired = self._repair_leading_brace(candidate)
        if repaired is not None:
            return repaired

        return {
            "parse_error": "Could not extract JSON from AI response",
            "raw_content": content,
        }

    def _strip_fences(self, content: str) -> str:
        """
        Drop markdown fences. The gateway eats the first token of every reply, so
        the opening ``` may be missing and leave a bare `json` marker behind.
        """
        text = self._FENCE_RE.sub("", content).strip()
        if text.lower().startswith("json"):
            text = text[len("json"):].lstrip()
        return text

    def _repair_leading_brace(self, candidate: str) -> dict | None:
        """
        Last resort for when the eaten first token was the opening brace itself
        (`{` and `{"` are both single tokens). Restore it and see if that parses.
        """
        stripped = candidate.strip()
        if not stripped.endswith("}"):
            return None

        for prefix in ('{"', "{"):
            try:
                return json.loads(prefix + stripped)
            except json.JSONDecodeError:
                continue
        return None
