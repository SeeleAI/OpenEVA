from __future__ import annotations

from typing import Dict, List


DEFAULT_QUESTION = "Describe this 3D object in detail."


def build_mesh_chat_messages(
    *,
    chat_history: List[Dict[str, str]],
    point_patch_text: str,
    max_history_turns: int,
) -> List[Dict[str, str]]:
    user_indices = [idx for idx, message in enumerate(chat_history) if message.get("role") == "user"]
    if not user_indices:
        raise ValueError("chat_history must contain at least one user message.")

    first_user_index = user_indices[0]
    first_user_content = str(chat_history[first_user_index].get("content") or "").strip()
    first_message = {
        "role": "user",
        "content": f"{point_patch_text}\n{first_user_content}",
    }
    remaining = chat_history[first_user_index + 1 :]
    if max_history_turns > 0:
        remaining = remaining[-int(max_history_turns) * 2 :]
    return [first_message] + [
        {"role": str(message.get("role")), "content": str(message.get("content") or "")}
        for message in remaining
        if message.get("role") in {"user", "assistant"} and str(message.get("content") or "").strip()
    ]


__all__ = ["DEFAULT_QUESTION", "build_mesh_chat_messages"]
