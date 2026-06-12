from __future__ import annotations

import ast
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_BASE_URL = "https://antchat.alipay.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_KEY_ENV = "ANTCHAT_API_KEY"
DEFAULT_API_KEY_FILE = "/home/public/minzhi/code/llm_judge_pressure_risk_sessions.py"

# Masked placeholder only. Prefer ANTCHAT_API_KEY for live runs.
api_key = "XXXXX"


def is_placeholder_api_key(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    return any(marker in value for marker in ("XXXXX", "YOUR_", "your_", "PLACEHOLDER", "<", ">", "*"))


def load_api_key_from_python_file(file_path: str, variable_name: str = "API_KEY") -> str:
    if not file_path:
        return ""
    path = Path(file_path)
    if not path.exists():
        return ""

    candidate_names = [variable_name]
    for fallback_name in ("API_KEY", "api_key", "apikey"):
        if fallback_name not in candidate_names:
            candidate_names.append(fallback_name)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in candidate_names:
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                value = node.value.value.strip()
                if not is_placeholder_api_key(value):
                    return value
    return ""


def resolve_api_key(
    explicit_api_key: str = "",
    env_name: str = DEFAULT_API_KEY_ENV,
    api_key_file: str = DEFAULT_API_KEY_FILE,
    api_key_var: str = "API_KEY",
) -> str:
    candidates = [
        explicit_api_key,
        os.environ.get(env_name, ""),
        load_api_key_from_python_file(api_key_file, api_key_var),
        api_key,
    ]
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and not is_placeholder_api_key(candidate):
            return candidate
    return ""


def extract_model_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content", "") if isinstance(message, dict) else ""
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


class RemoteChatClient:
    def __init__(
        self,
        *,
        api_base_url: str = DEFAULT_API_BASE_URL,
        api_key_value: str = "",
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
        use_env_proxy: bool = False,
        request_sleep: float = 0.0,
    ) -> None:
        self.api_base_url = api_base_url
        self.api_key_value = api_key_value
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_env_proxy = use_env_proxy
        self.request_sleep = request_sleep

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        if not self.api_key_value:
            raise ValueError("Missing API key. Set ANTCHAT_API_KEY or pass --api-key.")

        endpoint = self.api_base_url.rstrip("/") + "/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "enable_thinking": False,
        }

        session = requests.Session()
        session.trust_env = self.use_env_proxy
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            trace_id = str(uuid.uuid4())
            try:
                response = session.post(
                    endpoint,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key_value}",
                        "SOFA-TraceId": trace_id,
                        "SOFA-RpcId": "0.1",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                text = extract_model_text(response.json())
                if not text.strip():
                    raise RuntimeError("model returned empty content")
                if self.request_sleep > 0:
                    time.sleep(self.request_sleep)
                return text
            except (requests.RequestException, json.JSONDecodeError, RuntimeError) as exc:
                response_text = ""
                status_code = None
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    status_code = exc.response.status_code
                    response_text = exc.response.text[:1000]
                last_error = f"{type(exc).__name__}: {exc}"
                if response_text:
                    last_error += f"; response: {response_text}"
                if attempt < self.max_retries:
                    time.sleep(min(20 * attempt, 90) if status_code == 429 else min(2 * attempt, 8))
        raise RuntimeError(last_error or "model call failed")
