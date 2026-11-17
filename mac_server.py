"""Experimental Apple Silicon HTTP server for the MemT-1.0-Nano HF checkpoint.

The official MemT SGLang server is CUDA-only. This server uses the model's
published Transformers implementation and PyTorch MPS instead.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = "MemTensor/MemT-1.0-Nano"
PREFILL_CHUNK_SIZE = 512
logger = logging.getLogger(__name__)
MODEL_DIR = Path(os.environ.get("MEMT_MODEL_DIR", "")).expanduser()
STATE_DIR = Path(os.environ.get("MEMT_STATE_DIRECTORY", "memt-state")).expanduser()
TOOL_PATTERN = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^<>\s]+)>\s*"
    r"(?P<arguments>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
PARAMETER_PATTERN = re.compile(
    r"<parameter=(?P<name>[^<>\s]+)>\n?(?P<value>.*?)\n?</parameter>",
    re.DOTALL,
)


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    stop: str | list[str] | None = None


class SessionCreate(BaseModel):
    session_id: str | None = None
    system_prompt: str | None = None
    state: str | None = None
    tools: list[dict[str, Any]] | None = None


class SessionChat(BaseModel):
    query: str
    enable_memory: bool = True
    enable_thinking: bool = False
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)


class ToolResults(BaseModel):
    results: list[dict[str, Any]]


class Runtime:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.sessions: dict[str, Any] = {}
        self.lock = threading.RLock()

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable. Run on an Apple Silicon Mac outside a sandbox.")
        if not MODEL_DIR.is_dir() or not (MODEL_DIR / "model.safetensors.index.json").is_file():
            raise RuntimeError(f"Complete local checkpoint not found at {MODEL_DIR}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="mps",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).eval()
        gc.collect()
        torch.mps.empty_cache()
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def generate(self, request: ChatRequest) -> dict[str, Any]:
        import torch

        if request.model != MODEL_ID:
            raise HTTPException(404, f"Unknown model: {request.model}")
        if not request.messages or any(
            message.get("role") not in {"system", "user", "assistant", "tool"}
            or not isinstance(message.get("content"), str)
            for message in request.messages
        ):
            raise HTTPException(400, "messages must have supported roles and text content")
        options: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            options.update(temperature=request.temperature, top_p=request.top_p)
        template_options: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        if request.tools:
            template_options["tools"] = request.tools
        with self.lock, torch.inference_mode():
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            inputs = self.tokenizer.apply_chat_template(request.messages, **template_options)
            inputs = inputs.to("mps")
            if isinstance(inputs, torch.Tensor):
                input_ids = inputs
                input_tokens = int(input_ids.shape[-1])
                if input_tokens > PREFILL_CHUNK_SIZE:
                    options["prefill_chunk_size"] = PREFILL_CHUNK_SIZE
                logger.info("MemT prompt tokens=%d, prefill chunk=%s", input_tokens,
                            options.get("prefill_chunk_size", "none"))
                output = self.model.generate(inputs, **options)
            else:
                input_ids = inputs["input_ids"]
                input_tokens = int(input_ids.shape[-1])
                if input_tokens > PREFILL_CHUNK_SIZE:
                    options["prefill_chunk_size"] = PREFILL_CHUNK_SIZE
                logger.info("MemT prompt tokens=%d, prefill chunk=%s", input_tokens,
                            options.get("prefill_chunk_size", "none"))
                output = self.model.generate(**inputs, **options)
            generated = output[0, input_ids.shape[-1] :].detach().cpu().tolist()
            content = self.tokenizer.decode(generated, skip_special_tokens=True)
            output_tokens = len(generated)
        stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
        for stop in stops:
            if stop and stop in content:
                content = content.split(stop, 1)[0]
        tool_calls = parse_tool_calls(content, request.tools or [])
        if tool_calls:
            content = TOOL_PATTERN.sub("", content).strip() or None
        return {
            "content": content,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": (
                "tool_calls"
                if tool_calls
                else "length"
                if output_tokens >= request.max_tokens
                and generated[-1] != self.tokenizer.eos_token_id
                else "stop"
            ),
        }


def parse_tool_calls(text: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {})
        for tool in tools
        if tool.get("type", "function") == "function" and "function" in tool
    }
    calls = []
    for match in TOOL_PATTERN.finditer(text):
        name = match.group("name")
        if name not in schemas:
            continue
        properties = schemas[name].get("properties", {})
        arguments: dict[str, Any] = {}
        for parameter in PARAMETER_PATTERN.finditer(match.group("arguments")):
            key, value = parameter.group("name"), parameter.group("value")
            if properties.get(key, {}).get("type") != "string":
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            arguments[key] = value
        calls.append({
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        })
    return calls


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) or value in {".", ".."}:
        raise HTTPException(400, "Invalid session or snapshot name")
    return value


runtime = Runtime()
app = FastAPI(title="MemT-1.0-Nano Mac Server")


@app.on_event("startup")
def startup() -> None:
    runtime.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_ID, "device": "mps"}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> Any:
    result = runtime.generate(request)
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    message: dict[str, Any] = {"role": "assistant", "content": result["content"]}
    if result["tool_calls"]:
        message["tool_calls"] = result["tool_calls"]
    if not request.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": message, "finish_reason": result["finish_reason"]}],
            "usage": {
                "prompt_tokens": result["input_tokens"],
                "completion_tokens": result["output_tokens"],
                "total_tokens": result["input_tokens"] + result["output_tokens"],
            },
        }

    def events():
        base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
        delta: dict[str, Any] = {"role": "assistant"}
        if result["content"] is not None:
            delta["content"] = result["content"]
        if result["tool_calls"]:
            delta["tool_calls"] = [
                {"index": index, **call}
                for index, call in enumerate(result["tool_calls"])
            ]
        first = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}]}
        yield "data: " + json.dumps(first) + "\n\n"
        yield "data: " + json.dumps(last) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/memt/sessions")
def create_session(request: SessionCreate) -> dict[str, str]:
    session_id = safe_name(request.session_id or uuid.uuid4().hex)
    state = STATE_DIR / safe_name(request.state) if request.state else None
    if state is not None and not state.is_file():
        raise HTTPException(404, "Snapshot not found")
    with runtime.lock:
        if session_id in runtime.sessions:
            raise HTTPException(409, "Session already exists")
        runtime.sessions[session_id] = runtime.model.create_session(
            runtime.tokenizer, system_prompt=request.system_prompt,
            tools=request.tools, state=state
        )
    return {"session_id": session_id}


@app.post("/memt/sessions/{session_id}/chat")
def session_chat(session_id: str, request: SessionChat) -> dict[str, Any]:
    session = runtime.sessions.get(safe_name(session_id))
    if session is None:
        raise HTTPException(404, "Session not found")
    options: dict[str, Any] = {"max_new_tokens": request.max_tokens, "do_sample": request.temperature > 0}
    if request.temperature > 0:
        options.update(temperature=request.temperature, top_p=request.top_p)
    with runtime.lock:
        return session.chat(
            request.query,
            enable_memory=request.enable_memory,
            enable_thinking=request.enable_thinking,
            return_dict=True,
            **options,
        )


@app.post("/memt/sessions/{session_id}/tool-results")
def session_tool_results(session_id: str, request: ToolResults) -> dict[str, Any]:
    session = runtime.sessions.get(safe_name(session_id))
    if session is None:
        raise HTTPException(404, "Session not found")
    with runtime.lock:
        return session.submit_tool_results(request.results)


@app.post("/memt/sessions/{session_id}/save")
def save_session(session_id: str) -> dict[str, str]:
    session = runtime.sessions.get(safe_name(session_id))
    if session is None:
        raise HTTPException(404, "Session not found")
    with runtime.lock:
        result = session.save_state(STATE_DIR / f"{session_id}.safetensors", overwrite=True)
    return {"path": result["path"]}


@app.delete("/memt/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    with runtime.lock:
        session = runtime.sessions.pop(safe_name(session_id), None)
        if session is None:
            raise HTTPException(404, "Session not found")
        session.close()
    return {"deleted": True}
