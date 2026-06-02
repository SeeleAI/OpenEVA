#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eva01_instruct.conversation import DEFAULT_QUESTION
from eva01_instruct.generation import MeshInput, OpenEVARuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the OpenEVA-Instruct Gradio chat UI.")
    parser.add_argument("--checkpoint", default="checkpoints/openeva-instruct-lora")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None, choices=[None, "none", "auto"])
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-history-turns", type=int, default=4)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    import gradio as gr

    args = parse_args()
    runtime = OpenEVARuntime(
        checkpoint_dir=args.checkpoint,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        pointnum=args.pointnum,
        max_length=args.max_length,
    )

    def load_mesh(mesh_path: str | None) -> Tuple[str, MeshInput | None, List[Dict[str, str]], List[Dict[str, str]]]:
        if not mesh_path:
            return "No mesh selected.", None, [], []
        mesh_input = runtime.load_mesh_or_point_cloud(mesh_path, pointnum=args.pointnum)
        return f"Loaded {Path(mesh_path).name} as {mesh_input.object_id}.", mesh_input, [], []

    def respond(
        message: str,
        chat_messages: List[Tuple[str, str]],
        mesh_input: MeshInput | None,
        history: List[Dict[str, str]],
    ) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
        question = str(message or "").strip() or DEFAULT_QUESTION
        if mesh_input is None:
            chat_messages = list(chat_messages or [])
            chat_messages.append((question, "Load a mesh before chatting."))
            return chat_messages, list(history or [])
        response, new_history = runtime.answer(
            mesh_input=mesh_input,
            question=question,
            history=list(history or []),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_history_turns=args.max_history_turns,
        )
        chat_messages = list(chat_messages or [])
        chat_messages.append((question, response))
        return chat_messages, new_history

    def reset_chat() -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
        return [], []

    examples = sorted((REPO_ROOT / "examples" / "meshes").glob("*.glb"))[:2]
    with gr.Blocks(title="OpenEVA-Instruct") as demo:
        gr.Markdown("# OpenEVA-Instruct")
        with gr.Row():
            with gr.Column(scale=1):
                mesh = gr.Model3D(label="Mesh")
                upload = gr.File(label="Mesh or point cloud", file_types=[".glb", ".gltf", ".obj", ".ply", ".stl", ".npy", ".npz"])
                status = gr.Textbox(label="Status", interactive=False)
                load_button = gr.Button("Load")
                reset_button = gr.Button("Reset Chat")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat")
                prompt = gr.Textbox(label="Message", value=DEFAULT_QUESTION)
                send = gr.Button("Send")
        mesh_state = gr.State(None)
        history_state = gr.State([])

        def sync_preview(file_obj):
            if file_obj is None:
                return None
            return file_obj.name

        upload.change(sync_preview, inputs=upload, outputs=mesh)
        load_button.click(lambda f: f.name if f is not None else None, inputs=upload, outputs=mesh).then(
            load_mesh,
            inputs=mesh,
            outputs=[status, mesh_state, history_state, chatbot],
        )
        send.click(respond, inputs=[prompt, chatbot, mesh_state, history_state], outputs=[chatbot, history_state])
        reset_button.click(reset_chat, outputs=[chatbot, history_state])
        if examples:
            gr.Examples(examples=[[str(path)] for path in examples], inputs=upload)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
