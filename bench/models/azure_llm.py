from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from ..prompts import (SYSTEM_COT, SYSTEM_NOCOT, build_user_prompt,
                       parse_response)
from .base import PredictionRequest, PredictionResult

try:
    from azure.ai.inference.aio import ChatCompletionsClient
    from azure.ai.inference.models import (SystemMessage, UserMessage,
                                           TextContentItem, ImageContentItem,
                                           ImageUrl)
    from azure.core.credentials import AzureKeyCredential
    _HAVE_AZURE = True
except ImportError:
    _HAVE_AZURE = False

def _resolve_endpoint_and_key(model_name: str) -> tuple[str, str]:
    upper = model_name.upper().replace("-", "_").replace(".", "_")
    ep = os.getenv(f"AZURE_{upper}_ENDPOINT") or os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
    key = os.getenv(f"AZURE_{upper}_KEY") or os.getenv("AZURE_AI_FOUNDRY_KEY")
    if not ep or not key:
        raise RuntimeError(
            f"Missing endpoint/key for model {model_name!r}. Set "
            f"AZURE_AI_FOUNDRY_ENDPOINT and AZURE_AI_FOUNDRY_KEY in .env, or "
            f"the per-model AZURE_{upper}_ENDPOINT / AZURE_{upper}_KEY."
        )
    return ep, key

class AzureFoundryPredictor:
    is_async = True
    uses_modality = True
    uses_prompting = True
    uses_history = False

    def __init__(self, name: str, deployment: str, *, vision: bool = False,
                 concurrency: int = 6, request_timeout: float = 120.0,
                 max_retries: int = 4, api_version: str = "2024-08-01-preview"):
        if not _HAVE_AZURE:
            raise RuntimeError(
                "azure-ai-inference is not installed. Run "
                "`pip install -r requirements.txt`."
            )
        self.name = name
        self.deployment = deployment
        self.vision = vision
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.api_version = api_version
        self._sem = asyncio.Semaphore(concurrency)
        self._client: Optional[ChatCompletionsClient] = None

    async def _ensure_client(self):
        if self._client is None:
            ep, key = _resolve_endpoint_and_key(self.name)
            self._client = ChatCompletionsClient(
                endpoint=ep,
                credential=AzureKeyCredential(key),
                api_version=self.api_version,
            )

    async def aclose(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def apredict(self, req: PredictionRequest) -> PredictionResult:
        await self._ensure_client()
        cell = req.cell
        used_cot = (cell.prompting == "cot")
        system_msg = SYSTEM_COT if used_cot else SYSTEM_NOCOT
        user_text = build_user_prompt(req, k=cell.k, modality=cell.modality)

        modality = cell.modality
        if modality in ("images", "images_coords"):
            if not self.vision:
                return PredictionResult(
                    pred_theta=[float("nan")] * cell.k,
                    pred_omega=[float("nan")] * cell.k,
                    success=False,
                    error=f"model {self.name} not configured for vision modality",
                )
            if req.image_b64 is None:
                return PredictionResult(
                    pred_theta=[float("nan")] * cell.k,
                    pred_omega=[float("nan")] * cell.k,
                    success=False, error="no image supplied for vision modality",
                )
            content_items = [
                TextContentItem(text=user_text),
                ImageContentItem(image_url=ImageUrl(
                    url=f"data:image/png;base64,{req.image_b64}")),
            ]
            user_msg = UserMessage(content=content_items)
        else:
            user_msg = UserMessage(content=user_text)

        messages = [SystemMessage(content=system_msg), user_msg]

        last_err: Optional[BaseException] = None
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    resp = await asyncio.wait_for(
                        self._client.complete(
                            messages=messages,
                            model=self.deployment,
                            temperature=0.0,
                            max_tokens=2048 if used_cot else 256,
                        ),
                        timeout=self.request_timeout,
                    )
                    latency = time.perf_counter() - t0
                    text = resp.choices[0].message.content or ""
                    usage = getattr(resp, "usage", None)
                    p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                    c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                    try:
                        theta, omega, cot = parse_response(text, cell.k, used_cot)
                    except Exception as pe:
                        return PredictionResult(
                            pred_theta=[float("nan")] * cell.k,
                            pred_omega=[float("nan")] * cell.k,
                            raw_response=text, latency_s=latency,
                            prompt_tokens=p_tok, completion_tokens=c_tok,
                            success=False, error=f"parse: {pe}",
                        )
                    return PredictionResult(
                        pred_theta=theta, pred_omega=omega,
                        raw_response=text, cot_text=cot,
                        latency_s=latency,
                        prompt_tokens=p_tok, completion_tokens=c_tok,
                        success=True,
                    )
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        backoff = min(2 ** attempt + 0.5, 30.0)
                        await asyncio.sleep(backoff)
                    else:
                        break

        return PredictionResult(
            pred_theta=[float("nan")] * cell.k,
            pred_omega=[float("nan")] * cell.k,
            success=False, error=f"api: {last_err!r}",
        )
