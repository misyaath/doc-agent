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
    """
    Vision Config.

    Purpose:
        Defines VisionConfig in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.

    Attributes:
        ollama_url (str): Declared data field for this class.
        vision_model (str): Declared data field for this class.
        timeout_seconds (int): Declared data field for this class.
        max_image_side (int): Declared data field for this class.
        temperature (float): Declared data field for this class.
        num_predict (int): Declared data field for this class.
        num_ctx (int): Declared data field for this class.
    """

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    vision_model: str = os.getenv("VISION_MODEL", "llama3.2-vision:11b")
    timeout_seconds: int = 900
    max_image_side: int = 1400
    temperature: float = 0.0
    num_predict: int = 900
    num_ctx: int = 4096


class PromptBuilder(Protocol):
    """
    Prompt Builder.

    Purpose:
        Defines PromptBuilder in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def build(self, caption: str | None = None) -> str:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to PromptBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            caption (str | None): Input value for the caption parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside PromptBuilder so related code remains
                cohesive and testable.
        """
        ...


class ImageEncoder(Protocol):
    """
    Image Encoder.

    Purpose:
        Defines ImageEncoder in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def encode(self, image_path: str | Path) -> str:
        """
        Encode.

        Purpose:
            Implements encode for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ImageEncoder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_path (str | Path): Input value for the image path parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside ImageEncoder so related code remains
                cohesive and testable.
        """
        ...


class VisionClient(Protocol):
    """
    Vision Client.

    Purpose:
        Defines VisionClient in the document extraction pipeline that normalizes PDFs,
            enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def classify(self, image_base64: str, prompt: str) -> dict[str, Any]:
        """
        Classify.

        Purpose:
            Implements classify for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionClient; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_base64 (str): Input value for the image base64 parameter.
            prompt (str): Prompt text sent to the agent or language model.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionClient so related code remains
                cohesive and testable.
        """
        ...


class JsonPostProcessor(Protocol):
    """
    Json Post Processor.

    Purpose:
        Defines JsonPostProcessor in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process.

        Purpose:
            Implements process for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to JsonPostProcessor; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            data (dict[str, Any]): Input value for the data parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside JsonPostProcessor so related code remains
                cohesive and testable.
        """
        ...


class FigurePromptBuilder:
    """
    Figure Prompt Builder.

    Purpose:
        Defines FigurePromptBuilder in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def build(self, caption: str | None = None) -> str:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to FigurePromptBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            caption (str | None): Input value for the caption parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside FigurePromptBuilder so related code remains
                cohesive and testable.
        """
        return FIGURE_PROMPT_TEMPLATE.format(caption=caption or "")


class TablePromptBuilder:
    """
    Table Prompt Builder.

    Purpose:
        Defines TablePromptBuilder in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def build(self, caption: str | None = None) -> str:
        """
        Build.

        Purpose:
            Implements build for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to TablePromptBuilder; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            caption (str | None): Input value for the caption parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside TablePromptBuilder so related code remains
                cohesive and testable.
        """
        return TABLE_PROMPT_TEMPLATE.format(caption=caption or "")


class ResizedJpegBase64Encoder:
    """
    Resized Jpeg Base64 Encoder.

    Purpose:
        Defines ResizedJpegBase64Encoder in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, max_side: int = 1400) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ResizedJpegBase64Encoder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            max_side (int): Input value for the max side parameter.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside ResizedJpegBase64Encoder so related code
                remains cohesive and testable.
        """
        self._max_side = max_side

    def encode(self, image_path: str | Path) -> str:
        """
        Encode.

        Purpose:
            Implements encode for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to ResizedJpegBase64Encoder; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_path (str | Path): Input value for the image path parameter.
        Returns:
            str: Result produced by the operation.
        Why Added:
            Centralizes this behavior inside ResizedJpegBase64Encoder so related code
                remains cohesive and testable.
        """
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self._max_side, self._max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class OllamaVisionClient:
    """
    Ollama Vision Client.

    Purpose:
        Defines OllamaVisionClient in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(self, config: VisionConfig) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to OllamaVisionClient; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (VisionConfig): Configuration object controlling this component.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside OllamaVisionClient so related code remains
                cohesive and testable.
        """
        self._config = config

    def classify(self, image_base64: str, prompt: str) -> dict[str, Any]:
        """
        Classify.

        Purpose:
            Implements classify for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to OllamaVisionClient; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_base64 (str): Input value for the image base64 parameter.
            prompt (str): Prompt text sent to the agent or language model.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside OllamaVisionClient so related code remains
                cohesive and testable.
        """
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
    """
    Basic Json Parser.

    Purpose:
        Defines BasicJsonParser in the document extraction pipeline that normalizes
            PDFs, enriches visual content, builds RAG units, and indexes data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def parse(self, raw_model_output: str) -> dict[str, Any]:
        """
        Parse.

        Purpose:
            Implements parse for the document extraction pipeline that normalizes PDFs,
                enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to BasicJsonParser; uses that class state and dependencies when
                available.
        Args:
            self (Self): Current instance that owns the operation state.
            raw_model_output (str): Input value for the raw model output parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside BasicJsonParser so related code remains
                cohesive and testable.
        """
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
                f"Vision model returned invalid JSON. Error: {e}. Raw output preview: {raw_model_output[:1000]}"
            ) from e


class TableJsonPostProcessor:
    """
    Table Json Post Processor.

    Purpose:
        Defines TableJsonPostProcessor in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process.

        Purpose:
            Implements process for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to TableJsonPostProcessor; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            data (dict[str, Any]): Input value for the data parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside TableJsonPostProcessor so related code
                remains cohesive and testable.
        """
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
            "visible_text_long_summarycaption_summary",
            "rag_search_text",
        ]:
            if isinstance(data.get(key), str):
                data[key] = data[key].replace("…", "").strip()

        return data


class VisionAnalysisService:
    """
    Vision Analysis Service.

    Purpose:
        Defines VisionAnalysisService in the document extraction pipeline that
            normalizes PDFs, enriches visual content, builds RAG units, and indexes
            data.
    Why Added:
        Keeps this responsibility explicit so callers can depend on a named,
        documented component instead of duplicating the same logic elsewhere.
    """

    def __init__(
        self,
        config: VisionConfig | None = None,
        encoder: ImageEncoder | None = None,
        client: VisionClient | None = None,
    ) -> None:
        """
        Initialize the object with its required dependencies.

        Purpose:
            Implements __init__ for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionAnalysisService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            config (VisionConfig | None): Configuration object controlling this
                component.
            encoder (ImageEncoder | None): Input value for the encoder parameter.
            client (VisionClient | None): External service client used to call the
                backing service.
        Returns:
            None: Performs work through side effects and does not return a value.
        Why Added:
            Centralizes this behavior inside VisionAnalysisService so related code
                remains cohesive and testable.
        """
        self._config = config or VisionConfig()
        self._encoder = encoder or ResizedJpegBase64Encoder(max_side=self._config.max_image_side)
        self._client = client or OllamaVisionClient(config=self._config)
        self._parser = BasicJsonParser()
        self._figure_prompt_builder = FigurePromptBuilder()
        self._table_prompt_builder = TablePromptBuilder()
        self._table_post_processor = TableJsonPostProcessor()

    def analyze_figure(self, image_path: str | Path, caption: str | None = None) -> dict[str, Any]:
        """
        Analyze figure.

        Purpose:
            Implements analyze_figure for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisionAnalysisService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_path (str | Path): Input value for the image path parameter.
            caption (str | None): Input value for the caption parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionAnalysisService so related code
                remains cohesive and testable.
        """
        prompt = self._figure_prompt_builder.build(caption=caption)
        return self._analyze(image_path=image_path, prompt=prompt, post_processor=None)

    def analyze_table(self, image_path: str | Path, caption: str | None = None) -> dict[str, Any]:
        """
        Analyze table.

        Purpose:
            Implements analyze_table for the document extraction pipeline that
                normalizes PDFs, enriches visual content, builds RAG units, and indexes
                data.
        Class:
            Belongs to VisionAnalysisService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_path (str | Path): Input value for the image path parameter.
            caption (str | None): Input value for the caption parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionAnalysisService so related code
                remains cohesive and testable.
        """
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
        """
        Analyze.

        Purpose:
            Implements _analyze for the document extraction pipeline that normalizes
                PDFs, enriches visual content, builds RAG units, and indexes data.
        Class:
            Belongs to VisionAnalysisService; uses that class state and dependencies
                when available.
        Args:
            self (Self): Current instance that owns the operation state.
            image_path (str | Path): Input value for the image path parameter.
            prompt (str): Prompt text sent to the agent or language model.
            post_processor (JsonPostProcessor | None): Input value for the post
                processor parameter.
        Returns:
            dict[str, Any]: Structured data produced by the operation.
        Why Added:
            Centralizes this behavior inside VisionAnalysisService so related code
                remains cohesive and testable.
        """
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
