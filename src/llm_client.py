"""
src/llm_client.py - Gemini LLM Client using the current google-genai SDK.

Provides robust JSON completion with markdown fence stripping and deterministic
offline fallback stubs when GOOGLE_API_KEY is unset or unavailable.
"""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Wrapper for Google GenAI SDK (google-genai).
    Falls back gracefully to offline stub mode when no API key is provided.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model = model
        
        # Check environment or load from local .env
        if not api_key and not os.environ.get("GOOGLE_API_KEY"):
            env_file = Path(".env")
            if env_file.exists():
                try:
                    with open(env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            clean_line = line.strip()
                            if clean_line.startswith("GOOGLE_API_KEY="):
                                os.environ["GOOGLE_API_KEY"] = clean_line.split("=", 1)[1].strip("\"' ")
                                break
                except Exception:
                    pass

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self.is_offline = not bool(self.api_key)

        if not self.is_offline:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized GenAI client with model: {self.model}")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize GenAI client ({e}). Defaulting to offline stub mode."
                )
                self.is_offline = True
        else:
            logger.info("GOOGLE_API_KEY not set. Running in deterministic offline stub mode.")

    def complete_json(self, system: str, user: str, max_tokens: int = 500, max_retries: int = 3) -> Dict[str, Any]:
        """
        Calls Gemini with system instructions and user prompt, returning parsed JSON dictionary.
        Returns {"_offline_stub": True} if running offline.
        """
        if self.is_offline or self.client is None:
            return {"_offline_stub": True}

        import time

        for attempt in range(max_retries):
            try:
                config = {
                    "system_instruction": system,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                }
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=config,
                )

                raw_text = response.text or ""
                return self._clean_and_parse_json(raw_text)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Rate limit hit (429). Waiting {wait_time}s before retry (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Gemini API invocation error: {e}. Falling back to offline stub.")
                return {"_offline_stub": True, "error": str(e)}

    @staticmethod
    def _clean_and_parse_json(text: str) -> Dict[str, Any]:
        """Strip markdown code fences and parse JSON payload."""
        cleaned = text.strip()
        # Remove ```json ... ``` code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: attempt to find the first balanced JSON object in the text
            match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse valid JSON from text: {text[:100]}...")
            return {"_offline_stub": True, "parse_error": True, "raw_text": text}
