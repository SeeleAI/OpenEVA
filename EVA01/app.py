#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eva01 import EVA01ForConditionalGeneration, EVA01Processor


DEFAULT_MODEL = "SEELE-AI/EVA01-2B-Instruct"
DEFAULT_QUESTION = "Describe this 3D object in detail."
DEFAULT_SEED = 20260615


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the EVA01 mesh chat app.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--base-model", default=None, help="Optional local Qwen3VL base path for LoRA checkpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    chat_state = {"format": "messages"}

    def answer(mesh_path: str | None, message: str, history: list[dict[str, str]] | None):
        if not mesh_path:
            return history or [], "Load a GLB mesh first."
        prompt = (message or DEFAULT_QUESTION).strip()
        messages = [{"role": "user", "content": [{"type": "mesh", "mesh": mesh_path}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        set_seed(args.seed)
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        chat = list(history or [])
        if chat_state["format"] == "messages":
            chat.append({"role": "user", "content": prompt})
            chat.append({"role": "assistant", "content": text})
        else:
            chat.append((prompt, text))
        return chat, ""

    with gr.Blocks(title="EVA01") as demo:
        gr.Markdown("# EVA01")
        with gr.Row():
            with gr.Column(scale=1):
                mesh = gr.Model3D(label="Mesh")
                upload = gr.File(label="Upload GLB", file_types=[".glb"])
                prompt = gr.Textbox(label="Message", value=DEFAULT_QUESTION)
                send = gr.Button("Send", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                try:
                    chatbot = gr.Chatbot(label="Chat", type="messages")
                except TypeError:
                    chat_state["format"] = "tuples"
                    chatbot = gr.Chatbot(label="Chat")
        upload.change(lambda f: f.name if f else None, inputs=upload, outputs=mesh)
        send.click(answer, inputs=[mesh, prompt, chatbot], outputs=[chatbot, status])
        if examples:
            gr.Examples(examples=examples, inputs=[upload, prompt])

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
