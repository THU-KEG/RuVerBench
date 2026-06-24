import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from cache import Cache

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from zai import ZhipuAiClient
except Exception:
    ZhipuAiClient = None

def _normalize_base_url(base_url: str) -> str:
    """Normalize /chat/completions paths to /v1 style base URLs."""
    if not base_url:
        return base_url
    # Defensive cleanup: copied env values may include hidden control chars/newlines.
    url = re.sub(r"[\x00-\x1f\x7f]", "", str(base_url)).strip()
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "value" in item:
                    parts.append(str(item["value"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        for key in ("value", "text", "content"):
            if key in content:
                return str(content[key])
        return json.dumps(content, ensure_ascii=False)
    return "" if content is None else str(content)




# VERIBENCH_VLLM_FIXED20_SAMPLING_PATCH_V1
def _veribench_env_flag(name: str, default: str = "") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _veribench_env_int(name: str, default=None):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except Exception:
        raise ValueError(f"{name} must be an integer, got {value!r}")


def _veribench_env_float(name: str, default=None):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(str(value).strip())
    except Exception:
        raise ValueError(f"{name} must be a float, got {value!r}")


def _veribench_sampling_params(max_tokens=8192, temperature=0.0) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "max_tokens": _veribench_env_int("VERIBENCH_JUDGE_MAX_NEW_TOKENS", max_tokens),
        "temperature": _veribench_env_float("VERIBENCH_JUDGE_TEMPERATURE", temperature),
    }
    top_p = _veribench_env_float("VERIBENCH_JUDGE_TOP_P", None)
    if top_p is not None:
        params["top_p"] = top_p
    top_k = _veribench_env_int("VERIBENCH_JUDGE_TOP_K", None)
    if top_k is not None:
        params["top_k"] = top_k
    return params


def _veribench_apply_openai_sampling(create_kwargs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    create_kwargs["max_tokens"] = params["max_tokens"]
    create_kwargs["temperature"] = params["temperature"]
    if "top_p" in params:
        create_kwargs["top_p"] = params["top_p"]
    if "top_k" in params:
        extra_body = create_kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        extra_body["top_k"] = params["top_k"]
        create_kwargs["extra_body"] = extra_body
    return create_kwargs


def _veribench_cache_enabled(params: Dict[str, Any]) -> bool:
    if _veribench_env_flag("VERIBENCH_DISABLE_JUDGE_CACHE", "0"):
        return False
    if params.get("temperature") != 0.0:
        return False
    return not any(k in params for k in ("top_p", "top_k")) and os.getenv("VERIBENCH_JUDGE_MAX_NEW_TOKENS") in {None, ""}


def _veribench_cache_key(prompt: str, params: Dict[str, Any]) -> str:
    if _veribench_cache_enabled(params):
        return prompt
    return "[sampling_params]" + json.dumps(params, sort_keys=True, ensure_ascii=False) + "\n" + prompt


class APIModel:
    def __init__(self, cache_url, base_url, model_name, api_key="EMPTY"):
        base_url = _normalize_base_url(base_url)
        if "glm" in model_name.lower() and base_url == "":
            if ZhipuAiClient is None:
                raise ImportError("Missing dependency `zai` for GLM backend.")
            self.client = ZhipuAiClient(api_key=api_key)
        else:
            if not base_url:
                raise ValueError(
                    "JUDGE_BASE_URL must be set to an OpenAI-compatible /v1 endpoint "
                    "for public judge runs."
                )
            if OpenAI is None:
                raise ImportError("Missing dependency `openai` for OpenAI-compatible backend.")
            trust_env_raw = os.getenv("DETECTOR_HTTP_TRUST_ENV", "").strip().lower()
            # Default off for robustness in controlled server runs; opt in via env if needed.
            trust_env = trust_env_raw in {"1", "true", "yes", "on"}
            try:
                http_client = httpx.Client(
                    base_url=base_url,
                    follow_redirects=True,
                    trust_env=trust_env,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to init HTTP client. base_url={base_url!r}, trust_env={trust_env}. "
                    "If using local vLLM, set DETECTOR_HTTP_TRUST_ENV=0."
                ) from e
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
            )

        self.model_name = model_name
        self.base_url = base_url
        self.cache = Cache(cache_url)

    def generate(self, query, max_tokens=8192, temperature=0.0, response_format: Optional[dict] = None):
        params = _veribench_sampling_params(max_tokens=max_tokens, temperature=temperature)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        cache_key = _veribench_cache_key(query, params)
        if _veribench_cache_enabled(params):
            response = self.cache.check_prompt(cache_key)
        else:
            response = None

        if response is None:
            max_retries = 1
            retry_count = 0
            while retry_count < max_retries:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                    if "glm" in self.model_name.lower() and self.base_url == "":
                        chat_completion = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            thinking={"type": "enabled"},
                        )
                        if chat_completion.choices[0].finish_reason != "stop":
                            max_tokens = max_tokens + 2048
                            raise Exception(
                                f"Model stopped with reason: {chat_completion.choices[0].finish_reason}: {max_tokens}"
                            )
                        response = chat_completion.choices[0].message.content
                    else:
                        create_kwargs = _veribench_apply_openai_sampling({
                            "messages": messages,
                            "model": self.model_name,
                        }, params)
                        if response_format is not None:
                            create_kwargs["response_format"] = response_format
                        chat_completion = self.client.chat.completions.create(**create_kwargs)
                        if chat_completion.choices[0].finish_reason != "stop":
                            max_tokens = max_tokens + 2048
                            raise Exception(
                                f"Model stopped with reason: {chat_completion.choices[0].finish_reason}: {max_tokens}"
                            )
                        response = chat_completion.choices[0].message.content

                    if _veribench_cache_enabled(params):
                        self.cache.save_prompt(cache_key, response)
                        self.cache.maybe_flush(threshold=1)
                    break
                except Exception as e:
                    print(f"Attempt {retry_count + 1} failed: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        print("All retries failed")
                        response = None
        return response

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        """Best-effort normalize SDK usage objects into plain dict."""
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump") and callable(getattr(usage, "model_dump")):
            try:
                return usage.model_dump()
            except Exception:
                pass
        data = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
            "input_tokens",
            "output_tokens",
            "cache_input_tokens",
            "cached_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            if hasattr(usage, key):
                data[key] = getattr(usage, key)
        return data

    def generate_chat_with_metadata(self, messages, max_tokens=8192, temperature=0.0, response_format: Optional[dict] = None):
        params = _veribench_sampling_params(max_tokens=max_tokens, temperature=temperature)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        cache_key = _veribench_cache_key(_content_to_text(messages[-1]["content"]), params)
        metadata = {"usage": {}, "cache_hit": False}
        if _veribench_cache_enabled(params):
            response = self.cache.check_prompt(cache_key)
            if response is not None:
                metadata["cache_hit"] = True
        else:
            response = None

        if response is None:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if "glm" in self.model_name.lower() and self.base_url == "":
                        chat_completion = self.client.chat.completions.create(
                            messages=messages,
                            model=self.model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        response = chat_completion.choices[0].message.content
                        metadata["usage"] = self._usage_to_dict(getattr(chat_completion, "usage", None))
                    else:
                        create_kwargs = _veribench_apply_openai_sampling({
                            "messages": messages,
                            "model": self.model_name,
                        }, params)
                        if response_format is not None:
                            create_kwargs["response_format"] = response_format
                        chat_completion = self.client.chat.completions.create(**create_kwargs)
                        response = chat_completion.choices[0].message.content
                        metadata["usage"] = self._usage_to_dict(getattr(chat_completion, "usage", None))

                    if _veribench_cache_enabled(params):
                        self.cache.save_prompt(cache_key, response)
                        self.cache.maybe_flush()
                    break
                except Exception as e:
                    print(f"Attempt {retry_count + 1} failed: {e}")
                    retry_count += 1
                    if retry_count == max_retries:
                        print("All retries failed")
                        response = None
        return {"response": response, "usage": metadata["usage"], "cache_hit": metadata["cache_hit"]}

    def generate_chat(self, messages, max_tokens=8192, temperature=0.0, response_format: Optional[dict] = None):
        result = self.generate_chat_with_metadata(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        return result["response"]

    def save_cache(self):
        self.cache.flush()

    def add_n(self):
        return self.cache.add_n

    def close(self):
        self.cache.close()
