"""One-off converter: Cursor/Claude-Code session log -> bandits chat-json shape.

Not a bandits adapter. This is a throwaway script for one specific file format
(Anthropic-style content blocks, parentId chains, no tool results) that isn't
common enough yet to earn a real ingest/ adapter. Run it once per file you want
to try, then ingest the output with `bandits ingest ... --source chat-json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _text(content_blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in content_blocks if b.get("type") == "text")


def _tool_calls(content_blocks: list[dict]) -> list[dict]:
    return [
        {"id": b["id"], "function": {"name": b["name"], "arguments": b.get("input", {})}}
        for b in content_blocks
        if b.get("type") == "tool_use"
    ]


def convert(path: Path) -> list[dict]:
    messages: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "message":
            continue
        msg = record["message"]
        blocks = msg.get("content", [])
        entry: dict = {"role": msg["role"], "content": _text(blocks) or None}
        calls = _tool_calls(blocks)
        if calls:
            entry["tool_calls"] = calls
        messages.append(entry)
    return messages


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.write_text(json.dumps(convert(src), indent=2))
    print(f"wrote {dst}")
