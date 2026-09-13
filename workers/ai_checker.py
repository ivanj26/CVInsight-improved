"""
AIChecker — async OpenCode AI-generated content detector.

Accepts pre-extracted plain text and returns a structured verdict plus the
full raw response for auditability.

Instead of calling a remote OpenAI-compatible gateway (OmniRouter/TokenRouter),
this checker drives a one-shot OpenCode server session, mirroring
``LLMService._generate_content_with_opencode`` in ``cvinsight/core/llm_service.py``:
create a session, post a single message, read the returned text parts, then
delete the session.
"""

import json
import logging
import os
import re

import httpx

logger = logging.getLogger("ai_checker")


_SYSTEM_PROMPT = (
    "You are a helpful and trusted AI Checker to distinguish between human and "
    "AI-generated content."
)


def _build_user_prompt(text: str) -> str:
    return (
        f"Please carefully validate this file content [FILE START]\n{text}\n[FILE END].\n"
        "Give me a likelihood score from 0 to 100 of this file being AI generated and explain your reasoning under 720words in the `reasoning` field.\n"
        "If the likelihood score is above 65, please set the json property `is_ai_generated` to true, otherwise set it to false:\n\n"
        "Reply with ONLY a ```json fenced code block containing exactly this JSON format:\n"
        '{"likelihood_score": 0, "reasoning": "...", "is_ai_generated": false}'
    )


class AIChecker:
    _FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
    _JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

    def __init__(
        self,
        opencode_url: str | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._opencode_url = (opencode_url or os.environ.get("OPENCODE_URL", "http://localhost:4096")).rstrip("/")
        self._provider_id = provider_id or os.environ.get("OPENCODE_PROVIDER_ID")
        self._model_id = model_id or os.environ.get("OPENCODE_MODEL_ID")
        self._timeout = timeout if timeout is not None else float(os.environ.get("LLM_REQUEST_TIMEOUT", "60"))

        if not self._provider_id or not self._model_id:
            raise ValueError(
                "OPENCODE_PROVIDER_ID and OPENCODE_MODEL_ID are required for AIChecker."
            )

    async def check(self, text: str) -> dict:
        """
        Submit text to OpenCode and return:
            {
              "parsed": {"likelihood_score": int, "reasoning": str, "is_ai_generated": bool},
              "raw":    <full OpenCode response payload>
            }
        """
        prompt = f"{_SYSTEM_PROMPT}\n\n{_build_user_prompt(text)}"

        raw_payload = await self._request_opencode(prompt)

        info = raw_payload.get("info", {}) if isinstance(raw_payload, dict) else {}
        parts = raw_payload.get("parts", []) if isinstance(raw_payload, dict) else []
        raw_content = "".join(
            part.get("text", "") for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        )

        parsed = self._parse_response(raw_content)

        tokens = info.get("tokens", {}) if isinstance(info, dict) else {}
        reasoning = info.get("reasoning", {}) if isinstance(info, dict) else {}
        prompt_tokens = int(tokens.get("input", 0) or 0)
        completion_tokens = int(tokens.get("output", 0) or 0)
        total_tokens = int(tokens.get("total", 0) or 0) or (prompt_tokens + completion_tokens)

        raw = {
            "provider": "opencode",
            "session_id": info.get("sessionID") if isinstance(info, dict) else None,
            "model_id": self._model_id,
            "content": raw_content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "is_estimated": not bool(total_tokens),
            },
            "reasoning_tokens": int(reasoning.get("tokens", 0) or 0) if isinstance(reasoning, dict) else 0,
            "raw_response": raw_payload,
        }

        return {
            "parsed": parsed,
            "raw": raw,
        }

    async def _request_opencode(self, prompt: str) -> dict:
        """Create a one-shot OpenCode session, send the prompt, and clean up."""
        session_id = None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                session_response = await client.post(f"{self._opencode_url}/session", json={})
                session_response.raise_for_status()
                session_id = session_response.json().get("id")
                if not session_id:
                    raise ValueError("OpenCode did not return a session id")

                response = await client.post(
                    f"{self._opencode_url}/session/{session_id}/message",
                    json={
                        "model": {
                            "providerID": self._provider_id,
                            "modelID": self._model_id,
                        },
                        "parts": [{"type": "text", "text": prompt}],
                    },
                )
                response.raise_for_status()
                payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError("OpenCode returned an unexpected response payload")
            return payload
        finally:
            if session_id:
                try:
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        await client.delete(f"{self._opencode_url}/session/{session_id}")
                except Exception:
                    logger.warning("Unable to delete OpenCode session %s", session_id, exc_info=True)

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
