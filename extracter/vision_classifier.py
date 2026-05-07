from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
from PIL import Image

from .prompts import FIGURE_PROMPT_TEMPLATE, TABLE_PROMPT_TEMPLATE


@dataclass(frozen=True)
class VisionConfig:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    vision_model: str = os.getenv("VISION_MODEL", "llama3.2-vision:11b")
    timeout_seconds: int = 900
    max_image_side: int = 1400
    temperature: float = 0.0
    num_predict: int = 900
    num_ctx: int = 4096


class PromptBuilder(Protocol):
    def build(self, caption: str | None = None) -> str:
        ...


class ImageEncoder(Protocol):
    def encode(self, image_path: str | Path) -> str:
        ...


class VisionClient(Protocol):
    def classify(self, image_base64: str, prompt: str) -> dict[str, Any]:
        ...


class JsonPostProcessor(Protocol):
    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        ...


class FigurePromptBuilder:
    def build(self, caption: str | None = None) -> str:
        return FIGURE_PROMPT_TEMPLATE.format(caption=caption or "")


class TablePromptBuilder:
    def build(self, caption: str | None = None) -> str:
        return TABLE_PROMPT_TEMPLATE.format(caption=caption or "")


class ResizedJpegBase64Encoder:
    def __init__(self, max_side: int = 1400) -> None:
        self._max_side = max_side

    def encode(self, image_path: str | Path) -> str:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self._max_side, self._max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OllamaVisionClient:
    def __init__(self, config: VisionConfig) -> None:
        self._config = config

    def classify(self, image_base64: str, prompt: str) -> dict[str, Any]:
        response = requests.post(
            f"{self._config.ollama_url}/api/chat",
            json={
                "model": self._config.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64],
                    }
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": self._config.temperature,
                    "num_predict": self._config.num_predict,
                    "num_ctx": self._config.num_ctx,
                },
            },
            timeout=self._config.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return {"raw_model_output": content}


class BasicJsonParser:
    def parse(self, raw_model_output: str) -> dict[str, Any]:
        text = raw_model_output.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.replace("### JSON Output", "")
        text = text.strip()

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:500]}")

        json_text = match.group(0)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                "Vision model returned invalid JSON. "
                f"Error: {e}. "
                f"Raw output preview: {raw_model_output[:1000]}"
            ) from e


class TableJsonPostProcessor:
    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        data.setdefault("table_type", "other")
        data.setdefault("short_description", "")
        data.setdefault("columns_summary", "")
        data.setdefault("key_findings", [])
        data.setdefault("visible_text_summary", "")
        data.setdefault("visible_text_long_summary", "")
        data.setdefault("caption_summary", "")
        data.setdefault("rag_search_text", "")
        data.setdefault("rag_keywords", [])
        data.setdefault("should_index_for_rag", True)

        if not isinstance(data["key_findings"], list):
            data["key_findings"] = []

        if not isinstance(data["rag_keywords"], list):
            data["rag_keywords"] = []

        if not isinstance(data["should_index_for_rag"], bool):
            data["should_index_for_rag"] = True

        for key in [
            "short_description",
            "columns_summary",
            "visible_text_summary",
            "visible_text_long_summary"
            "caption_summary",
            "rag_search_text",
        ]:
            if isinstance(data.get(key), str):
                data[key] = data[key].replace("…", "").strip()

        return data


class VisionAnalysisService:
    def __init__(
            self,
            config: VisionConfig | None = None,
            encoder: ImageEncoder | None = None,
            client: VisionClient | None = None,
    ) -> None:
        self._config = config or VisionConfig()
        self._encoder = encoder or ResizedJpegBase64Encoder(max_side=self._config.max_image_side)
        self._client = client or OllamaVisionClient(config=self._config)
        self._parser = BasicJsonParser()
        self._figure_prompt_builder = FigurePromptBuilder()
        self._table_prompt_builder = TablePromptBuilder()
        self._table_post_processor = TableJsonPostProcessor()

    def analyze_figure(self, image_path: str | Path, caption: str | None = None) -> dict[str, Any]:
        prompt = self._figure_prompt_builder.build(caption=caption)
        return self._analyze(image_path=image_path, prompt=prompt, post_processor=None)

    def analyze_table(self, image_path: str | Path, caption: str | None = None) -> dict[str, Any]:
        prompt = self._table_prompt_builder.build(caption=caption)
        return self._analyze(
            image_path=image_path,
            prompt=prompt,
            post_processor=self._table_post_processor,
        )

    def _analyze(
            self,
            image_path: str | Path,
            prompt: str,
            post_processor: JsonPostProcessor | None,
    ) -> dict[str, Any]:
        image_base64 = self._encoder.encode(image_path)
        raw = self._client.classify(image_base64=image_base64, prompt=prompt)
        raw_text = raw.get("raw_model_output", "")

        try:
            parsed = self._parser.parse(raw_text)
        except Exception as e:
            parsed = {
                "parse_error": str(e),
                "raw_model_output_preview": raw_text[:1000],
                "rag_search_text": "",
                "should_index_for_rag": False,
            }

        if post_processor and not parsed.get("parse_error"):
            parsed = post_processor.process(parsed)

        return {
            "raw_model_output": raw_text,
            "parsed": parsed,
        }
