#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from threading import Thread
from typing import Any

import numpy as np
import torch
from transformers import TextIteratorStreamer

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eva01 import EVA01ForConditionalGeneration, EVA01Processor


DEFAULT_MODEL = "SEELE-AI/EVA01-2B-Instruct"
DEFAULT_QUESTION = "Describe this 3D object in detail."
DEFAULT_SEED = 20260615
DEFAULT_MAX_HISTORY_TURNS = 6
VALID_CHAT_ROLES = {"user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the EVA01 mesh chat app.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--base-model", default=None, help="Optional local Qwen3VL base path for LoRA checkpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-history-turns", type=int, default=DEFAULT_MAX_HISTORY_TURNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_device_map(value: str | None) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def normalize_mesh_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name") or value.get("orig_name")
        return normalize_mesh_path(candidate)
    if hasattr(value, "name"):
        return normalize_mesh_path(getattr(value, "name"))
    text = str(value).strip()
    return text or None


def chat_message(role: str, content: str) -> dict[str, str]:
    resolved_role = role if role in VALID_CHAT_ROLES else "assistant"
    return {"role": resolved_role, "content": str(content or "")}


def sanitize_ui_messages(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    messages: list[dict[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            role = str(item.get("role", "assistant"))
            content = item.get("content", "")
        else:
            role = str(getattr(item, "role", "assistant"))
            content = getattr(item, "content", "")
        if role in VALID_CHAT_ROLES:
            messages.append(chat_message(role, str(content or "")))
    return messages


def sanitize_model_messages(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        if role not in VALID_CHAT_ROLES:
            continue
        content = item.get("content", "")
        if isinstance(content, list):
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": str(content or "")})
    return messages


def trim_model_messages(messages: list[dict[str, Any]], max_history_turns: int) -> list[dict[str, Any]]:
    if not messages:
        return []
    if max_history_turns <= 0:
        return [messages[0]]
    keep_tail = max_history_turns * 2
    return [messages[0], *messages[1:][-keep_tail:]]


def reset_conversation(mesh_value: Any = None) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], str | None, str]:
    mesh_path = normalize_mesh_path(mesh_value)
    status = "Mesh loaded. Chat reset." if mesh_path else "Load a GLB mesh first."
    return [], [], [], mesh_path, status


def handle_upload(uploaded_file: Any) -> tuple[str | None, list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], str | None, str]:
    mesh_path = normalize_mesh_path(uploaded_file)
    chat, ui_state, model_state, mesh_state, status = reset_conversation(mesh_path)
    return mesh_path, chat, ui_state, model_state, mesh_state, status


def build_generation_messages(
    *,
    mesh_path: str,
    prompt: str,
    model_history: list[dict[str, Any]],
    max_history_turns: int,
) -> list[dict[str, Any]]:
    if not model_history:
        next_user = {
            "role": "user",
            "content": [
                {"type": "mesh", "mesh": mesh_path},
                {"type": "text", "text": prompt},
            ],
        }
        return [next_user]
    return trim_model_messages([*model_history, {"role": "user", "content": prompt}], max_history_turns=max_history_turns)


def main() -> None:
    import gradio as gr

    args = parse_args()
    set_seed(args.seed)
    dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    processor = EVA01Processor.from_pretrained(args.checkpoint)
    model = EVA01ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        device_map=normalize_device_map(args.device_map),
        base_model_name_or_path=args.base_model,
    )

    examples_dir = ROOT / "assets" / "examples"
    metadata_path = examples_dir / "gallery.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    examples = [[str(examples_dir / item["file"]), item.get("q2", DEFAULT_QUESTION)] for item in metadata]

    def answer(
        mesh_value: Any,
        message: str,
        ui_history: Any,
        model_history: Any,
        current_mesh: str | None,
    ):
        mesh_path = normalize_mesh_path(mesh_value)
        ui_messages = sanitize_ui_messages(ui_history)
        model_messages = sanitize_model_messages(model_history)
        if not mesh_path:
            yield ui_messages, ui_messages, model_messages, current_mesh, "Load a GLB mesh first."
            return

        if current_mesh != mesh_path:
            ui_messages = []
            model_messages = []

        prompt = (message or DEFAULT_QUESTION).strip() or DEFAULT_QUESTION
        generation_messages = build_generation_messages(
            mesh_path=mesh_path,
            prompt=prompt,
            model_history=model_messages,
            max_history_turns=args.max_history_turns,
        )
        ui_messages = [*ui_messages, chat_message("user", prompt), chat_message("assistant", "")]
        yield ui_messages, ui_messages, model_messages, mesh_path, "Generating..."

        inputs = processor.apply_chat_template(
            generation_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        generation_error: list[BaseException] = []

        def run_generation() -> None:
            try:
                set_seed(args.seed)
                model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                )
            except BaseException as exc:
                generation_error.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        thread = Thread(target=run_generation, daemon=True)
        thread.start()
        answer_text = ""
        for chunk in streamer:
            answer_text += chunk
            ui_messages[-1] = chat_message("assistant", answer_text)
            yield ui_messages, ui_messages, model_messages, mesh_path, "Generating..."
        thread.join()

        if generation_error:
            error_text = f"Error: {generation_error[0]}"
            ui_messages[-1] = chat_message("assistant", error_text)
            yield ui_messages, ui_messages, model_messages, mesh_path, error_text
            return

        final_text = answer_text.strip()
        ui_messages[-1] = chat_message("assistant", final_text)
        updated_model_messages = trim_model_messages(
            [*generation_messages, {"role": "assistant", "content": final_text}],
            max_history_turns=args.max_history_turns,
        )
        yield ui_messages, ui_messages, updated_model_messages, mesh_path, ""

    with gr.Blocks(title="EVA01") as demo:
        gr.Markdown("# EVA01")
        ui_state = gr.State([])
        model_state = gr.State([])
        mesh_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                mesh = gr.Model3D(label="Mesh")
                upload = gr.File(label="Upload GLB", file_types=[".glb"])
                prompt = gr.Textbox(label="Message", value=DEFAULT_QUESTION)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear Chat")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=520)

        upload.change(handle_upload, inputs=upload, outputs=[mesh, chatbot, ui_state, model_state, mesh_state, status])
        mesh.change(reset_conversation, inputs=mesh, outputs=[chatbot, ui_state, model_state, mesh_state, status])
        clear.click(reset_conversation, inputs=mesh, outputs=[chatbot, ui_state, model_state, mesh_state, status])
        send.click(
            answer,
            inputs=[mesh, prompt, ui_state, model_state, mesh_state],
            outputs=[chatbot, ui_state, model_state, mesh_state, status],
        )
        if examples:
            gr.Examples(examples=examples, inputs=[mesh, prompt])

    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
