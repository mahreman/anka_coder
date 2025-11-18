#!/usr/bin/env python
"""
Qwen3-Coder + Ollama ile otonom coder agent (v2).

Bu modül:
- Config'i okur
- Qwen3-Coder'a prompt atar
- JSON formatında dosya aksiyonları alır
- İstenirse workspace içinde dosyaları GERÇEKTEN yazar/siler (dry-run destekli)
- Özet string döndürür (web UI'de chat mesajı olarak gösterilecek)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from ollama import Client  # type: ignore


# ---------------------------------------------------------------------------
# Config yapısı
# ---------------------------------------------------------------------------

@dataclass
class ModelEndpoint:
    host: str
    model: str


@dataclass
class GeneralConfig:
    default_workspace: str
    temperature: float
    max_rounds: int
    max_files_in_tree: int
    max_preview_bytes_per_file: int
    dry_run_default: bool


@dataclass
class AppConfig:
    mode: str  # "cloud" veya "local"
    cloud: ModelEndpoint
    local: ModelEndpoint
    general: GeneralConfig


def load_config(config_path: Path) -> AppConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Ayar dosyası bulunamadı: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mode = str(raw.get("mode", "cloud")).lower()

    cloud_raw = raw.get("cloud", {})
    local_raw = raw.get("local", {})
    general_raw = raw.get("general", {})

    cloud = ModelEndpoint(
        host=str(cloud_raw.get("host", "http://localhost:11434")),
        model=str(cloud_raw.get("model", "qwen3-coder:480b-cloud")),
    )
    local = ModelEndpoint(
        host=str(local_raw.get("host", "http://localhost:11434")),
        model=str(local_raw.get("model", "qwen2.5-coder:32b")),
    )
    general = GeneralConfig(
        default_workspace=str(general_raw.get("default_workspace", "./workspace")),
        temperature=float(general_raw.get("temperature", 0.1)),
        max_rounds=int(general_raw.get("max_rounds", 1)),
        max_files_in_tree=int(general_raw.get("max_files_in_tree", 150)),
        max_preview_bytes_per_file=int(
            general_raw.get("max_preview_bytes_per_file", 2000)
        ),
        dry_run_default=bool(general_raw.get("dry_run_default", False)),
    )

    return AppConfig(mode=mode, cloud=cloud, local=local, general=general)


def select_endpoint(cfg: AppConfig, override_mode: Optional[str] = None) -> ModelEndpoint:
    mode = override_mode.lower() if override_mode else cfg.mode.lower()
    if mode == "local":
        return cfg.local
    return cfg.cloud


# ---------------------------------------------------------------------------
# Prompt & LLM çağrısı
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an autonomous coding agent similar to Claude Coder / OpenAI Codex.

You ONLY work inside a given workspace directory and you have FULL control over files:
- You can create, overwrite, and delete files.
- You must ALWAYS write full file contents (never patches, never diffs).
- Prefer the simplest, most robust solution.

### GOAL

The user will give you:
- A natural language task.
- The absolute path of the workspace directory.
- A file tree overview of the current workspace (relative paths).

You must:
1. Understand the task.
2. Decide the minimal set of files to create/update/delete to complete the task.
3. Output a JSON array describing the file operations.

### FILE OPERATIONS FORMAT

You MUST respond with a single Markdown JSON code block:

```json
[
  {
    "path": "relative/path/from/workspace.py",
    "action": "write",
    "description": "short explanation for this file",
    "content": "FULL FILE CONTENT HERE"
  },
  {
    "path": "relative/old_file_to_delete.py",
    "action": "delete",
    "description": "why this file is removed"
  }
]
```

Rules:

* "path" MUST be relative to the workspace root (no .., no absolute paths).
* "action" is one of: "write", "delete".
* For "write", you MUST include the full final file content in "content".
* For "delete", do NOT include "content".
* You MUST return valid JSON (no comments, no trailing commas).
* Outside the `json ... ` block you MUST NOT output anything (no explanation, no markdown, nothing).

If the task is not clear, make reasonable assumptions and still produce a JSON file plan.
"""

def get_workspace_overview(workspace: Path, max_files: int) -> str:
    """
    Workspace içindeki dosya ağacını kısa bir metin olarak çıkar.
    Modeller için bağlam verir.
    """
    items: List[str] = []
    for p in workspace.rglob("*"):
        if p.is_file():
            rel = p.relative_to(workspace)
            items.append(str(rel))
        if len(items) >= max_files:
            break
    if not items:
        return "(empty workspace)"
    return "\n".join(items)

def build_messages(task: str, workspace: Path, cfg: AppConfig) -> List[Dict[str, Any]]:
    file_tree = get_workspace_overview(workspace, cfg.general.max_files_in_tree)
    user_content = (
        "Workspace absolute path:\n"
        f"{str(workspace.resolve())}\n\n"
        "Workspace file tree (relative paths, truncated):\n"
        f"{file_tree}\n\n"
        "Task:\n"
        f"{task}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


def call_llm(
    endpoint: ModelEndpoint,
    messages: List[Dict[str, Any]],
    temperature: float,
) -> str:
    client = Client(host=endpoint.host)
    response = client.chat(
        model=endpoint.model,
        messages=messages,
        options={"temperature": float(temperature)},
    )
    # Ollama python client dict döndürür: {"message": {"content": "..."}}
    return response["message"]["content"]  # type: ignore[index]

# ---------------------------------------------------------------------------
# JSON blok ayıklama & dosya işlemleri
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> str:
    """
    Model çıktısından `json ... ` bloğunu çek.
    Eğer yoksa tüm metni JSON sanıp parse etmeyi deneriz.
    """
    start_token = "```json"
    end_token = "```"

    start_idx = text.find(start_token)
    if start_idx == -1:
        return text.strip()

    start_idx += len(start_token)
    end_idx = text.find(end_token, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()

    return text[start_idx:end_idx].strip()


def ensure_path_in_workspace(workspace: Path, target: Path) -> None:
    """
    path traversal engelle: workspace dışına çıkmasın.
    """
    workspace_resolved = workspace.resolve()
    target_resolved = target.resolve()
    if workspace_resolved == target_resolved:
        return
    if workspace_resolved not in target_resolved.parents:
        raise ValueError(f"Workspace dışı path reddedildi: {target_resolved}")

def apply_file_actions(actions: List[Dict[str, Any]], workspace: Path) -> List[str]:
    """
    Dosya işlemlerini uygular ve kısa log satırları döner.
    """
    logs: List[str] = []
    for action in actions:
        path_str = action.get("path")
        op = action.get("action")
        if not path_str or not op:
            logs.append(f"[UYARI] Geçersiz action (path/action yok): {action}")
            continue

        target_path = workspace / path_str
        ensure_path_in_workspace(workspace, target_path)

        if op == "write":
            content = action.get("content", "")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            logs.append(f"[WRITE] {path_str} ({len(content)} byte)")
        elif op == "delete":
            if target_path.exists():
                target_path.unlink()
                logs.append(f"[DELETE] {path_str}")
            else:
                logs.append(f"[SKIP-DELETE] {path_str} (zaten yok)")
        else:
            logs.append(f"[UYARI] Bilinmeyen action türü: {op} ({path_str})")

    return logs


def parse_actions(json_text: str) -> List[Dict[str, Any]]:
    data = json.loads(json_text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "files" in data and isinstance(data["files"], list):
        return data["files"]
    raise ValueError("Beklenen format bir JSON listesi veya {'files': [...]} olmalı.")

def summarize_actions(
    actions: List[Dict[str, Any]],
    logs: Optional[List[str]] = None,
    dry_run: bool = False,
) -> str:
    """
    Web chat'te gösterilecek kısa özet.
    """
    if not actions:
        return "Herhangi bir dosya değişikliği gerekmedi."

    write_count = 0
    delete_count = 0
    lines: List[str] = []

    for a in actions:
        path = a.get("path", "?")
        op = a.get("action", "?")
        desc = a.get("description", "")
        if op == "write":
            write_count += 1
            prefix = "YAZ"
        elif op == "delete":
            delete_count += 1
            prefix = "SİL"
        else:
            prefix = op.upper()
        if desc:
            lines.append(f"- [{prefix}] {path} → {desc}")
        else:
            lines.append(f"- [{prefix}] {path}")

    header = f"{write_count} dosya yazılacak, {delete_count} dosya silinecek."
    if not dry_run:
        header = f"{write_count} dosya yazıldı, {delete_count} dosya silindi."

    result = header + "\n" + "\n".join(lines)

    if logs:
        result += "\n\nLog:\n" + "\n".join(logs)

    if dry_run:
        result = "🟡 DRY-RUN (dosyalara dokunulmadı)\n\n" + result
    else:
        result = "🟢 UYGULANDI\n\n" + result

    return result


# ---------------------------------------------------------------------------
# Dışarı açılan ana fonksiyon
# ---------------------------------------------------------------------------

def run_agent_round(
    config: AppConfig,
    task: str,
    workspace: Path,
    mode_override: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Tek tur:
    - LLM'e git
    - JSON aksiyon listesi al
    - İstenirse workspace'e uygula
    - Aksiyon listesini ve insan okunur özeti döndür
    """
    workspace.mkdir(parents=True, exist_ok=True)
    endpoint = select_endpoint(config, mode_override)
    messages = build_messages(task, workspace, config)

    raw_output = call_llm(
        endpoint=endpoint,
        messages=messages,
        temperature=config.general.temperature,
    )

    json_text = extract_json_block(raw_output)

    # JSON ayrıştırma hatasında dosyalara dokunma
    try:
        actions = parse_actions(json_text)
    except Exception as e:
        summary = (
            "❌ Model çıktısı geçersiz JSON gibi görünüyor, dosyalara dokunulmadı.\n\n"
            f"Hata: {e}\n\n"
            "Ham çıktı (kısaltılmış):\n"
            f"{raw_output[:2000]}{'...' if len(raw_output) > 2000 else ''}"
        )
        return [], summary

    # Dry-run kararı
    if dry_run is None:
        dry_run_flag = config.general.dry_run_default
    else:
        dry_run_flag = dry_run

    logs: List[str] = []
    if not dry_run_flag:
        try:
            logs = apply_file_actions(actions, workspace)
        except Exception as e:
            summary = (
                "❌ Dosya işlemlerinde hata oluştu.\n\n"
                f"Hata: {e}"
            )
            return actions, summary

    summary = summarize_actions(actions, logs, dry_run=dry_run_flag)
    return actions, summary
